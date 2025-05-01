package hfdownloader

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"bytes"
	"context"
	"net"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/fatih/color"
	"github.com/schollz/progressbar/v3"
)

const (
	AgreementModelURL      = "https://huggingface.co/%s"
	AgreementDatasetURL    = "https://huggingface.co/datasets/%s"
	RawModelFileURL        = "https://huggingface.co/%s/raw/%s/%s"
	RawDatasetFileURL      = "https://huggingface.co/datasets/%s/raw/%s/%s"
	LfsModelResolverURL    = "https://huggingface.co/%s/resolve/%s/%s"
	LfsDatasetResolverURL  = "https://huggingface.co/datasets/%s/resolve/%s/%s"
	JsonModelsFileTreeURL  = "https://huggingface.co/api/models/%s/tree/%s/%s"
	JsonDatasetFileTreeURL = "https://huggingface.co/api/datasets/%s/tree/%s/%s"
	// Optimize for high-speed downloads
	streamBufferSize   = 256 * 1024 * 1024      // 256MB buffer
	multipartThreshold = 1024 * 1024 * 1024     // 1GB threshold
	chunkSize          = 256 * 1024 * 1024      // 256MB chunks
	maxConcurrent      = 16                     // 8 concurrent files
	maxPartsPerFile    = 32                     // 16 concurrent chunks per file
	bufferSize         = 128 * 1024 * 1024      // 128MB buffer
	maxRetries         = 3                      // Reduced retries for faster failure recovery
	retryDelay         = 500 * time.Millisecond // Shorter retry delay
)

var (
	infoColor      = color.New(color.FgGreen).SprintFunc()
	successColor   = color.New(color.FgHiGreen).SprintFunc()
	warningColor   = color.New(color.FgYellow).SprintFunc()
	errorColor     = color.New(color.FgRed).SprintFunc()
	NumConnections = 64 // Increased from 5 to 32
	RequiresAuth   = false
	AuthToken      = ""
)

type hfmodel struct {
	Type          string `json:"type"`
	Oid           string `json:"oid"`
	Size          int    `json:"size"`
	Path          string `json:"path"`
	LocalSize     int64
	NeedsDownload bool
	IsDirectory   bool
	IsLFS         bool

	AppendedPath    string
	SkipDownloading bool
	FilterSkip      bool
	DownloadLink    string
	Lfs             *hflfs `json:"lfs,omitempty"`
}

type hflfs struct {
	Oid_SHA265  string `json:"oid"` // in lfs, oid is sha256 of the file
	Size        int64  `json:"size"`
	PointerSize int    `json:"pointerSize"`
}

type R2Config struct {
	AccountID       string
	AccessKeyID     string
	AccessKeySecret string
	BucketName      string
	Region          string // Usually "auto" for R2
	Subfolder       string // Custom subfolder (e.g., "hf_dataset")
}

type uploadProgress struct {
	progress *progressbar.ProgressBar
	mu       sync.Mutex
}

type progressReader struct {
	reader   io.Reader
	progress *uploadProgress
}

func (r *progressReader) Read(p []byte) (n int, err error) {
	n, err = r.reader.Read(p)
	if r.progress != nil {
		r.progress.Add(int64(n))
	}
	return
}

func newProgressReader(reader io.Reader, progress *uploadProgress) io.Reader {
	return &progressReader{
		reader:   reader,
		progress: progress,
	}
}

func createProgressBar(total int64, filename string) *uploadProgress {
	bar := progressbar.NewOptions64(
		total,
		progressbar.OptionSetDescription(filename),
		progressbar.OptionShowBytes(true),
		progressbar.OptionSetWidth(30),
		progressbar.OptionThrottle(65*time.Millisecond),
		progressbar.OptionShowCount(),
		progressbar.OptionOnCompletion(func() {
			fmt.Printf("\n")
		}),
	)

	return &uploadProgress{
		progress: bar,
	}
}

func (p *uploadProgress) Add(n int64) {
	if p == nil || p.progress == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	_ = p.progress.Add64(n)
}

// Add this struct to store file metadata
type R2FileCache struct {
	files map[string]struct{}
	mu    sync.RWMutex
}

// Add this function to pre-fetch existing files
func buildR2Cache(ctx context.Context, r2cfg *R2Config, prefix string) (*R2FileCache, error) {
	client := createR2Client(ctx, *r2cfg)
	cache := &R2FileCache{
		files: make(map[string]struct{}),
	}

	input := &s3.ListObjectsV2Input{
		Bucket: aws.String(r2cfg.BucketName),
		Prefix: aws.String(prefix),
	}

	fmt.Printf("Building cache of existing files in R2...\n")
	start := time.Now()

	// Use paginator for large buckets
	paginator := s3.NewListObjectsV2Paginator(client, input)
	count := 0

	for paginator.HasMorePages() {
		page, err := paginator.NextPage(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to list objects: %v", err)
		}

		for _, obj := range page.Contents {
			cache.files[*obj.Key] = struct{}{}
			count++
		}

		if count%1000 == 0 {
			fmt.Printf("Cached %d files...\n", count)
		}
	}

	elapsed := time.Since(start)
	fmt.Printf("Cached %d files in %s\n", count, elapsed)
	return cache, nil
}

// Add method to check if file exists
func (c *R2FileCache) Exists(key string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	_, exists := c.files[key]
	return exists
}

func DownloadModel(ModelDatasetName string, AppendFilterToPath bool, SkipSHA bool, IsDataset bool, DestinationBasePath string, ModelBranch string, concurrentConnections int, token string, silentMode bool, r2cfg *R2Config, skipLocal bool, hfPrefix string) error {
	// Build cache of existing files
	ctx := context.Background()
	cache, err := buildR2Cache(ctx, r2cfg, r2cfg.Subfolder+"/")
	if err != nil {
		return fmt.Errorf("failed to build R2 cache: %v", err)
	}

	modelP := strings.Split(ModelDatasetName, ":")[0]
	modelPath := filepath.Join(DestinationBasePath, modelP)

	// Create R2 client for checking existing files
	// r2Client := createR2Client(ctx, *r2cfg)

	maxWorkers := 16
	jobs := make(chan hfmodel, maxWorkers)
	results := make(chan error, maxWorkers)
	var wg sync.WaitGroup
	var completedFiles atomic.Int32

	for i := 0; i < maxWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for file := range jobs {
				if file.IsDirectory || file.FilterSkip || file.Size <= 0 || file.Path == "" {
					completedFiles.Add(1)
					continue
				}
				if skipLocal && file.LocalSize > 0 {
					completedFiles.Add(1)
					continue
				}
				if file.SkipDownloading {
					completedFiles.Add(1)
					continue
				}
				if file.IsLFS {
					if !silentMode {
						fmt.Printf("Skipping LFS file %s\n", file.Path)
					}
					completedFiles.Add(1)
					continue
				}

				fmt.Printf("Worker %d: Processing file %s\n", workerID, file.Path)

				r2Key := fmt.Sprintf("%s/%s", r2cfg.Subfolder, strings.TrimPrefix(file.Path, fmt.Sprintf("%s/", hfPrefix)))

				// Fast cache lookup instead of HeadObject
				if cache.Exists(r2Key) {
					if !silentMode {
						fmt.Printf("Skipping %s - already exists in R2\n", r2Key)
					}
					completedFiles.Add(1)
					continue
				}

				downloadURL := fmt.Sprintf("https://huggingface.co/datasets/%s/resolve/%s/%s",
					ModelDatasetName,
					ModelBranch,
					file.Path,
				)

				fmt.Printf("Worker %d: Starting download of %s\n", workerID, file.Path)

				// Create request
				req, err := http.NewRequest("GET", downloadURL, nil)
				if err != nil {
					fmt.Printf("Error creating request for %s: %v\n", file.Path, err)
					results <- fmt.Errorf("failed to create request for %s: %v", file.Path, err)
					continue
				}

				if RequiresAuth {
					req.Header.Add("Authorization", "Bearer "+AuthToken)
				}
				req.Header.Add("User-Agent", "Mozilla/5.0")

				// Download file
				resp, err := http.DefaultClient.Do(req)
				if err != nil {
					fmt.Printf("Error downloading %s: %v\n", file.Path, err)
					results <- fmt.Errorf("failed to download %s: %v", file.Path, err)
					continue
				}


				if resp.StatusCode != http.StatusOK {
					resp.Body.Close()
					fmt.Printf("Error downloading %s: status %d\n", file.Path, resp.StatusCode)
					results <- fmt.Errorf("failed to download %s: status %d", file.Path, resp.StatusCode)
					continue
				}

				// Create progress bar
				progress := createProgressBar(int64(file.Size), filepath.Base(file.Path))

				// Upload to R2
				var uploadErr error
				if int64(file.Size) > multipartThreshold {
					uploadErr = streamMultipartToR2(ctx, *r2cfg, resp.Body, r2Key, int64(file.Size), progress)
				} else {
					uploadErr = streamSimpleToR2(ctx, *r2cfg, resp.Body, r2Key, int64(file.Size), progress)
				}
				resp.Body.Close()

				if uploadErr != nil {
					fmt.Printf("Error uploading %s: %v\n", file.Path, uploadErr)
					results <- fmt.Errorf("failed to upload %s: %v", file.Path, uploadErr)
					continue
				}

				// Verify parquet file
				if err := verifyParquetFile(ctx, r2cfg, r2Key, int64(file.Size)); err != nil {
					// Delete corrupted file
					client := createR2Client(ctx, *r2cfg)
					_, deleteErr := client.DeleteObject(ctx, &s3.DeleteObjectInput{
						Bucket: aws.String(r2cfg.BucketName),
						Key:    aws.String(r2Key),
					})
					if deleteErr != nil {
						fmt.Printf("Warning: Failed to delete corrupted file %s: %v\n", r2Key, deleteErr)
					}

					results <- fmt.Errorf("file verification failed for %s: %v", r2Key, err)
					continue
				}

				completedFiles.Add(1)
				fmt.Printf("✅ Worker %d: Successfully uploaded and verified %s\n", workerID, r2Key)
			}
		}(i)
	}

	// Process files function that checks cache before queueing
	processFiles := func(files []hfmodel) {
		var pendingFiles []hfmodel
		totalSize := int64(0)
		skippedSize := int64(0)
		skippedCount := 0

		// First, filter files that need to be processed
		for _, file := range files {
			if !file.IsDirectory && !file.FilterSkip && file.Size > 0 {
				r2Key := fmt.Sprintf("%s/%s", r2cfg.Subfolder, strings.TrimPrefix(file.Path, fmt.Sprintf("%s/", hfPrefix)))

				totalSize += int64(file.Size)

				if cache.Exists(r2Key) {
					skippedSize += int64(file.Size)
					skippedCount++
					continue
				}

				pendingFiles = append(pendingFiles, file)
			}
		}

		// Print summary
		if !silentMode {
			fmt.Printf("\n=== Processing Summary ===\n")
			fmt.Printf("Total files found: %d\n", len(files))
			fmt.Printf("Files already in R2: %d\n", skippedCount)
			fmt.Printf("Files to process: %d\n", len(pendingFiles))
			fmt.Printf("Total size: %s\n", formatSize(totalSize))
			fmt.Printf("Skipped size: %s\n", formatSize(skippedSize))
			fmt.Printf("Remaining size: %s\n\n", formatSize(totalSize-skippedSize))
		}

		// Queue only files that need processing
		for _, file := range pendingFiles {
			if !silentMode {
				fmt.Printf("Queueing: %s (%s)\n", file.Path, formatSize(int64(file.Size)))
			}
			jobs <- file
		}
	}

	// Start processing
	err = processHFFolderTree(modelPath, IsDataset, SkipSHA, ModelDatasetName, ModelBranch, "", silentMode, r2cfg, skipLocal, processFiles, hfPrefix)
	if err != nil {
		return fmt.Errorf("error processing file tree: %v", err)
	}

	// Close jobs and wait
	close(jobs)
	wg.Wait()
	close(results)

	// Check for errors
	var errors []error
	for err := range results {
		errors = append(errors, err)
	}

	if len(errors) > 0 {
		return fmt.Errorf("encountered errors: %v", errors)
	}

	return nil
}

func processHFFolderTree(modelPath string, IsDataset bool, SkipSHA bool, ModelDatasetName string, ModelBranch string, folderName string, silentMode bool, r2cfg *R2Config, skipLocal bool, processFiles func([]hfmodel), hfPrefix string) error {
	if !silentMode {
		fmt.Printf("🔍 Scanning: %s\n", folderName)
	}

	// Build the correct API URL
	var url string
	if IsDataset {
		if folderName == "" {
			url = fmt.Sprintf(JsonDatasetFileTreeURL, ModelDatasetName, ModelBranch, hfPrefix)
		} else {
			url = fmt.Sprintf(JsonDatasetFileTreeURL, ModelDatasetName, ModelBranch, folderName)
		}
	}

	if !silentMode {
		fmt.Printf("📡 API URL: %s\n", url)
	}

	// Make request and get files
	files, err := fetchFileList(url)
	if err != nil {
		return err
	}

	if !silentMode {
		fmt.Printf("📂 Found %d items in %s\n", len(files), folderName)
	}


	var parquetFiles []hfmodel
	for _, file := range files {
		if strings.HasSuffix(file.Path, ".parquet") && file.Size > 0 {
			if file.IsLFS {
				file.DownloadLink = fmt.Sprintf(LfsDatasetResolverURL, ModelDatasetName, ModelBranch, file.Path)
			} else {
				file.DownloadLink = fmt.Sprintf(RawDatasetFileURL, ModelDatasetName, ModelBranch, file.Path)
			}
			parquetFiles = append(parquetFiles, file)
		} else {
			if !silentMode {
				fmt.Printf("📁 Entering directory: %s\n", file.Path)
			}

			err := processHFFolderTree(modelPath, IsDataset, SkipSHA, ModelDatasetName, ModelBranch, file.Path, silentMode, r2cfg, skipLocal, processFiles, hfPrefix)
			if err != nil {
				fmt.Printf("⚠️ Error processing subdirectory %s: %v\n", file.Path, err)
				continue
			}
		}
	}

	if len(parquetFiles) > 0 {
		if !silentMode {
			fmt.Printf("📦 Processing %d parquet files from %s\n", len(parquetFiles), folderName)
		}
		processFiles(parquetFiles)
	}

	return nil
}

// Helper function to fetch and parse file list
func fetchFileList(url string) ([]hfmodel, error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	if RequiresAuth {
		req.Header.Add("Authorization", "Bearer "+AuthToken)
	}
	req.Header.Add("User-Agent", "Mozilla/5.0")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch file list: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("failed to fetch file list, status: %d, body: %s", resp.StatusCode, string(body))
	}

	var files []hfmodel
	if err := json.NewDecoder(resp.Body).Decode(&files); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}

	return files, nil
}

// Helper function for size formatting
func formatSize(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// ***********************************************   All the functions below generated by ChatGPT 3.5, and ChatGPT 4 , with some modifications ***********************************************
func IsValidModelName(modelName string) bool {
	pattern := `^[A-Za-z0-9_\-]+/[A-Za-z0-9\._\-]+$`
	match, _ := regexp.MatchString(pattern, modelName)
	return match
}

// Add parallel chunk downloading
func streamMultipartToR2(ctx context.Context, r2cfg R2Config, reader io.Reader, key string, contentLength int64, progress *uploadProgress) error {
	client := createR2Client(ctx, r2cfg)

	// First, check for and clean up any existing incomplete multipart uploads
	listResp, err := client.ListMultipartUploads(ctx, &s3.ListMultipartUploadsInput{
		Bucket: aws.String(r2cfg.BucketName),
		Prefix: aws.String(key),
	})
	if err == nil && listResp.Uploads != nil {
		for _, upload := range listResp.Uploads {
			_, err := client.AbortMultipartUpload(ctx, &s3.AbortMultipartUploadInput{
				Bucket:   aws.String(r2cfg.BucketName),
				Key:      upload.Key,
				UploadId: upload.UploadId,
			})
			if err != nil {
				fmt.Printf("Warning: Failed to abort incomplete upload for %s: %v\n", *upload.Key, err)
			}
		}
	}

	// Create new multipart upload
	resp, err := client.CreateMultipartUpload(ctx, &s3.CreateMultipartUploadInput{
		Bucket: aws.String(r2cfg.BucketName),
		Key:    aws.String(key),
	})
	if err != nil {
		return fmt.Errorf("failed to create multipart upload: %v", err)
	}
	uploadID := *resp.UploadId

	// Calculate optimal part size (minimum 5MB, maximum 5GB)
	partSize := contentLength / int64(maxPartsPerFile)
	if partSize < 5*1024*1024 {
		partSize = 5 * 1024 * 1024 // 5MB minimum
	}
	if partSize > 5*1024*1024*1024 {
		partSize = 5 * 1024 * 1024 * 1024 // 5GB maximum
	}

	// Create parts channel and results
	type partResult struct {
		Part types.CompletedPart
		Err  error
	}
	parts := make([]types.CompletedPart, 0)
	results := make(chan partResult, maxPartsPerFile)
	var wg sync.WaitGroup

	// Read and upload parts
	var partNum int32 = 1
	remainingBytes := contentLength

	for remainingBytes > 0 {
		size := partSize
		if remainingBytes < partSize {
			size = remainingBytes
		}

		buffer := make([]byte, size)
		n, err := io.ReadFull(reader, buffer)
		if err != nil && err != io.ErrUnexpectedEOF {
			// Abort upload on error
			_, abortErr := client.AbortMultipartUpload(ctx, &s3.AbortMultipartUploadInput{
				Bucket:   aws.String(r2cfg.BucketName),
				Key:      aws.String(key),
				UploadId: aws.String(uploadID),
			})
			if abortErr != nil {
				fmt.Printf("Warning: Failed to abort upload after error: %v\n", abortErr)
			}
			return fmt.Errorf("failed to read part %d: %v", partNum, err)
		}

		wg.Add(1)
		go func(num int32, buf []byte) {
			defer wg.Done()

			// Upload part
			partResp, err := client.UploadPart(ctx, &s3.UploadPartInput{
				Bucket:     aws.String(r2cfg.BucketName),
				Key:        aws.String(key),
				PartNumber: aws.Int32(num),
				UploadId:   aws.String(uploadID),
				Body:       bytes.NewReader(buf),
			})

			if err != nil {
				results <- partResult{Err: fmt.Errorf("failed to upload part %d: %v", num, err)}
				return
			}

			results <- partResult{
				Part: types.CompletedPart{
					PartNumber: aws.Int32(num),
					ETag:       partResp.ETag,
				},
			}

			if progress != nil {
				progress.Add(int64(len(buf)))
			}
		}(partNum, buffer[:n])

		partNum++
		remainingBytes -= int64(n)
	}

	// Wait for all parts to complete
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results
	var uploadErr error
	for result := range results {
		if result.Err != nil {
			uploadErr = result.Err
			break
		}
		parts = append(parts, result.Part)
	}

	if uploadErr != nil {
		// Abort upload on error
		_, abortErr := client.AbortMultipartUpload(ctx, &s3.AbortMultipartUploadInput{
			Bucket:   aws.String(r2cfg.BucketName),
			Key:      aws.String(key),
			UploadId: aws.String(uploadID),
		})
		if abortErr != nil {
			fmt.Printf("Warning: Failed to abort upload after error: %v\n", abortErr)
		}
		return uploadErr
	}

	// Sort parts by part number
	sort.Slice(parts, func(i, j int) bool {
		return *parts[i].PartNumber < *parts[j].PartNumber
	})

	// Complete upload
	_, err = client.CompleteMultipartUpload(ctx, &s3.CompleteMultipartUploadInput{
		Bucket:          aws.String(r2cfg.BucketName),
		Key:             aws.String(key),
		UploadId:        aws.String(uploadID),
		MultipartUpload: &types.CompletedMultipartUpload{Parts: parts},
	})

	return err
}

// Optimize S3 client configuration
func createR2Client(ctx context.Context, r2cfg R2Config) *s3.Client {
	r2Resolver := aws.EndpointResolverWithOptionsFunc(func(service, region string, options ...interface{}) (aws.Endpoint, error) {
		return aws.Endpoint{
			URL: fmt.Sprintf("https://%s.r2.cloudflarestorage.com", r2cfg.AccountID),
		}, nil
	})

	cfg, err := config.LoadDefaultConfig(ctx,
		config.WithEndpointResolverWithOptions(r2Resolver),
		config.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(
			r2cfg.AccessKeyID,
			r2cfg.AccessKeySecret,
			"",
		)),
		config.WithRegion(r2cfg.Region),
		// Add performance configurations
		config.WithHTTPClient(&http.Client{
			Transport: &http.Transport{
				MaxIdleConns:        256,
				MaxIdleConnsPerHost: 256,
				IdleConnTimeout:     30 * time.Second,
				DisableCompression:  true,
				MaxConnsPerHost:     256,
				WriteBufferSize:     64 * 1024,
				ReadBufferSize:      64 * 1024,
				DialContext: (&net.Dialer{
					Timeout:   10 * time.Second,
					KeepAlive: 30 * time.Second,
				}).DialContext,
			},
			Timeout: 30 * time.Minute,
		}),
	)

	if err != nil {
		panic(err)
	}

	return s3.NewFromConfig(cfg)
}

// Helper function for simple uploads
func streamSimpleToR2(ctx context.Context, r2cfg R2Config, reader io.Reader, key string, contentLength int64, progress *uploadProgress) error {
	// For parquet files, verify before upload
	if strings.HasSuffix(key, ".parquet") {
		// Create a temp file for verification
		tmpFile, err := os.CreateTemp("", "parquet-verify-*")
		if err != nil {
			return fmt.Errorf("failed to create temp file: %v", err)
		}
		defer os.Remove(tmpFile.Name())
		defer tmpFile.Close()

		// Copy data to temp file
		if _, err := io.Copy(tmpFile, reader); err != nil {
			return fmt.Errorf("failed to copy to temp file: %v", err)
		}

		// Verify parquet format
		if err := verifyLocalParquet(tmpFile.Name()); err != nil {
			return fmt.Errorf("invalid parquet file: %v", err)
		}

		// Reset file for upload
		if _, err := tmpFile.Seek(0, io.SeekStart); err != nil {
			return fmt.Errorf("failed to reset file: %v", err)
		}

		// Use temp file as reader
		reader = tmpFile
	}

	if progress == nil {
		progress = createProgressBar(contentLength, filepath.Base(key))
	}

	client := createR2Client(ctx, r2cfg)
	progressReader := newProgressReader(reader, progress)

	length := contentLength
	_, err := client.PutObject(ctx, &s3.PutObjectInput{
		Bucket:        aws.String(r2cfg.BucketName),
		Key:           aws.String(key),
		Body:          progressReader,
		ContentLength: &length,
	})

	if err != nil {
		return fmt.Errorf("upload failed: %v", err)
	}

	// Verify after upload
	if strings.HasSuffix(key, ".parquet") {
		if err := verifyParquetFile(ctx, &r2cfg, key, contentLength); err != nil {
			// Delete the failed upload
			_, delErr := client.DeleteObject(ctx, &s3.DeleteObjectInput{
				Bucket: aws.String(r2cfg.BucketName),
				Key:    aws.String(key),
			})
			if delErr != nil {
				fmt.Printf("Warning: Failed to delete invalid upload: %v\n", delErr)
			}
			return fmt.Errorf("post-upload verification failed: %v", err)
		}
	}

	return nil
}

func verifyParquetFile(ctx context.Context, r2cfg *R2Config, key string, expectedSize int64) error {
	client := createR2Client(ctx, *r2cfg)

	// Get first 4 bytes
	headerObj, err := client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(r2cfg.BucketName),
		Key:    aws.String(key),
		Range:  aws.String("bytes=0-3"),
	})
	if err != nil {
		return fmt.Errorf("failed to get header: %v", err)
	}
	defer headerObj.Body.Close()

	header := make([]byte, 4)
	if _, err := io.ReadFull(headerObj.Body, header); err != nil {
		return fmt.Errorf("failed to read header: %v", err)
	}

	// Get last 4 bytes
	footerObj, err := client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(r2cfg.BucketName),
		Key:    aws.String(key),
		Range:  aws.String(fmt.Sprintf("bytes=%d-%d", expectedSize-4, expectedSize-1)),
	})
	if err != nil {
		return fmt.Errorf("failed to get footer: %v", err)
	}
	defer footerObj.Body.Close()

	footer := make([]byte, 4)
	if _, err := io.ReadFull(footerObj.Body, footer); err != nil {
		return fmt.Errorf("failed to read footer: %v", err)
	}

	// Verify magic numbers
	expectedMagic := []byte("PAR1")
	if !bytes.Equal(header, expectedMagic) {
		return fmt.Errorf("invalid parquet header magic number")
	}
	if !bytes.Equal(footer, expectedMagic) {
		return fmt.Errorf("invalid parquet footer magic number")
	}

	return nil
}

// Add this function to verify local parquet file before upload
func verifyLocalParquet(filePath string) error {
	// Open file
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	// Read header
	header := make([]byte, 4)
	if _, err := io.ReadFull(file, header); err != nil {
		return fmt.Errorf("failed to read header: %v", err)
	}

	// Read footer (seek to end - 4)
	if _, err := file.Seek(-4, io.SeekEnd); err != nil {
		return fmt.Errorf("failed to seek to footer: %v", err)
	}

	footer := make([]byte, 4)
	if _, err := io.ReadFull(file, footer); err != nil {
		return fmt.Errorf("failed to read footer: %v", err)
	}

	// Verify magic numbers
	expectedMagic := []byte("PAR1")
	if !bytes.Equal(header, expectedMagic) {
		return fmt.Errorf("invalid parquet header magic number")
	}
	if !bytes.Equal(footer, expectedMagic) {
		return fmt.Errorf("invalid parquet footer magic number")
	}

	return nil
}

func CleanupCorruptedFiles(ctx context.Context, r2cfg *R2Config, prefix string, concurrency int) error {
	client := createR2Client(ctx, *r2cfg)
	var wg sync.WaitGroup
	jobs := make(chan *types.Object, concurrency*2) // buffered channel for efficiency
	var totalFiles, corruptedFiles int32

	// Worker function to process verification for each parquet file.
	worker := func(workerID int) {
		defer wg.Done()
		for obj := range jobs {
			if !strings.HasSuffix(*obj.Key, ".parquet") || *obj.Size < 8 {
				continue
			}

			fmt.Printf("[Worker %d] Checking file: %s (size: %s)\n", workerID, *obj.Key, formatSize(*obj.Size))
			err := verifyParquetFile(ctx, r2cfg, *obj.Key, *obj.Size)
			// If header/footer check passed, then perform full checksum validation if metadata is available
			if err == nil {
				// Retrieve object's metadata to get expected checksum
				head, headErr := client.HeadObject(ctx, &s3.HeadObjectInput{
					Bucket: aws.String(r2cfg.BucketName),
					Key:    obj.Key,
				})
				if headErr == nil {
					if expected, ok := head.Metadata["sha256"]; ok && expected != "" {
						fmt.Printf("[Worker %d] Verifying checksum for: %s (expected: %s)\n", workerID, *obj.Key, expected)
						err = verifyRemoteFileChecksum(ctx, r2cfg, *obj.Key, expected)
					} else {
						fmt.Printf("[Worker %d] No sha256 metadata for: %s, skipping checksum\n", workerID, *obj.Key)
					}
				} else {
					fmt.Printf("[Worker %d] Failed to retrieve metadata for: %s, skipping checksum (error: %v)\n", workerID, *obj.Key, headErr)
				}
			}

			atomic.AddInt32(&totalFiles, 1)
			if err != nil {
				atomic.AddInt32(&corruptedFiles, 1)
				fmt.Printf("[Worker %d] ❌ Corrupted file: %s, error: %v\n", workerID, *obj.Key, err)
				if strings.Contains(err.Error(), "invalid parquet") {
					_, delErr := client.DeleteObject(ctx, &s3.DeleteObjectInput{
						Bucket: aws.String(r2cfg.BucketName),
						Key:    obj.Key,
					})
					if delErr != nil {
						fmt.Printf("[Worker %d] Warning: Failed to delete file %s: %v\n", workerID, *obj.Key, delErr)
					} else {
						fmt.Printf("[Worker %d] Deleted corrupted file: %s\n", workerID, *obj.Key)
					}
				}
			} else {
				fmt.Printf("[Worker %d] ✅ Valid parquet file: %s\n", workerID, *obj.Key)
			}
		}
	}

	// Spawn the specified number of workers.
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go worker(i)
	}

	// List objects from R2 using the provided prefix.
	input := &s3.ListObjectsV2Input{
		Bucket: aws.String(r2cfg.BucketName),
		Prefix: aws.String(prefix),
	}
	paginator := s3.NewListObjectsV2Paginator(client, input)
	for paginator.HasMorePages() {
		page, err := paginator.NextPage(ctx)
		if err != nil {
			close(jobs)
			wg.Wait()
			return fmt.Errorf("failed to list objects: %v", err)
		}
		fmt.Printf("Retrieved %d objects with prefix %s\n", len(page.Contents), prefix)
		for _, obj := range page.Contents {
			// send pointer to a copy to avoid the type error
			o := obj
			jobs <- &o
		}
	}
	close(jobs)
	wg.Wait()

	fmt.Printf("\n=== Summary ===\n")
	fmt.Printf("Total parquet files checked: %d\n", totalFiles)
	fmt.Printf("Corrupted files found: %d\n", corruptedFiles)
	if totalFiles == 0 {
		fmt.Printf("Warning: No parquet files found! Verify bucket and prefix.\n")
	}
	fmt.Printf("Verification complete!\n")
	return nil
}

// Add this helper function to hfdownloader/hfdownloader.go
func verifyRemoteFileChecksum(ctx context.Context, r2cfg *R2Config, key string, expectedChecksum string) error {
	client := createR2Client(ctx, *r2cfg)
	obj, err := client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(r2cfg.BucketName),
		Key:    aws.String(key),
	})
	if err != nil {
		return fmt.Errorf("failed to download file for checksum: %v", err)
	}
	defer obj.Body.Close()

	hash := sha256.New()
	if _, err := io.Copy(hash, obj.Body); err != nil {
		return fmt.Errorf("failed to compute checksum: %v", err)
	}
	computed := hex.EncodeToString(hash.Sum(nil))
	if computed != expectedChecksum {
		return fmt.Errorf("checksum mismatch: computed %s, expected %s", computed, expectedChecksum)
	}
	return nil
}
