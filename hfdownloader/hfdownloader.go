package hfdownloader

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/fatih/color"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"context"
	"bytes"
	// "crypto/md5"
	// "encoding/base64"
	"github.com/schollz/progressbar/v3"
	// "bufio"
	"net"
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
	streamBufferSize    = 256 * 1024 * 1024  // 256MB buffer
	multipartThreshold  = 1024 * 1024 * 1024 // 1GB threshold
	chunkSize          = 256 * 1024 * 1024  // 256MB chunks
	maxConcurrent      = 8                   // 8 concurrent files
	maxPartsPerFile    = 16                  // 16 concurrent chunks per file
	bufferSize         = 128 * 1024 * 1024  // 128MB buffer
	maxRetries         = 3                    // Reduced retries for faster failure recovery
	retryDelay         = 500 * time.Millisecond // Shorter retry delay
)

var (
	infoColor      = color.New(color.FgGreen).SprintFunc()
	successColor   = color.New(color.FgHiGreen).SprintFunc()
	warningColor   = color.New(color.FgYellow).SprintFunc()
	errorColor     = color.New(color.FgRed).SprintFunc()
	NumConnections = 32  // Increased from 5 to 32
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
	BucketName     string
	Region         string // Usually "auto" for R2
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

func DownloadModel(ModelDatasetName string, AppendFilterToPath bool, SkipSHA bool, IsDataset bool, DestinationBasePath string, ModelBranch string, concurrentConnections int, token string, silentMode bool, r2cfg *R2Config, skipLocal bool) error {
	modelP := strings.Split(ModelDatasetName, ":")[0]
	modelPath := filepath.Join(DestinationBasePath, modelP)
	
	// Create R2 client for checking existing files
	ctx := context.Background()
	r2Client := createR2Client(ctx, *r2cfg)

	maxWorkers := 8
	jobs := make(chan hfmodel, maxWorkers)
	results := make(chan error, maxWorkers)
	var wg sync.WaitGroup
	var completedFiles atomic.Int32

	for i := 0; i < maxWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for file := range jobs {
				// Check if file already exists in R2
				r2Key := fmt.Sprintf("HuggingFaceFW_fineweb-edu-score-2/%s", strings.TrimPrefix(file.Path, "data/"))
				
				_, err := r2Client.HeadObject(ctx, &s3.HeadObjectInput{
					Bucket: aws.String(r2cfg.BucketName),
					Key:    aws.String(r2Key),
				})

				if err == nil {
					fmt.Printf("Skipping %s - already exists in R2\n", r2Key)
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
					results <- fmt.Errorf("failed to download %s: %v", file.Path, err)
					continue
				}

				if resp.StatusCode != http.StatusOK {
					resp.Body.Close()
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
					results <- fmt.Errorf("failed to upload %s: %v", file.Path, uploadErr)
					continue
				}

				completedFiles.Add(1)
				fmt.Printf("✅ Worker %d: Completed %s\n", workerID, r2Key)
			}
		}(i)
	}

	// Process files function
	processFiles := func(files []hfmodel) {
		for _, file := range files {
			if !file.IsDirectory && !file.FilterSkip && file.Size > 0 {
				// Add delay between queueing files
				time.Sleep(time.Second)
				fmt.Printf("Queueing: %s (%s)\n", file.Path, formatSize(int64(file.Size)))
				jobs <- file
			}
		}
	}

	// Process the file tree
	err := processHFFolderTree(modelPath, IsDataset, SkipSHA, ModelDatasetName, ModelBranch, "", silentMode, r2cfg, skipLocal, processFiles)
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

func processHFFolderTree(modelPath string, IsDataset bool, SkipSHA bool, ModelDatasetName string, ModelBranch string, folderName string, silentMode bool, r2cfg *R2Config, skipLocal bool, processFiles func([]hfmodel)) error {
	if !silentMode {
		fmt.Printf("🔍 Scanning: %s\n", folderName)
	}

	// Build the correct API URL
	var url string
	if IsDataset {
		if folderName == "" {
			url = fmt.Sprintf(JsonDatasetFileTreeURL, ModelDatasetName, ModelBranch, "data")
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

	// For each CC-MAIN directory, fetch its contents
	for _, file := range files {
		// Check if it's a CC-MAIN directory (they're reported as files but are actually directories)
		if strings.Contains(file.Path, "CC-MAIN-") {
			dirPath := file.Path
			if !silentMode {
				fmt.Printf("📁 Entering directory: %s\n", dirPath)
			}

			// Fetch contents of this CC-MAIN directory
			dirUrl := fmt.Sprintf(JsonDatasetFileTreeURL, ModelDatasetName, ModelBranch, dirPath)
			dirFiles, err := fetchFileList(dirUrl)
			if err != nil {
				fmt.Printf("⚠️ Error fetching directory %s: %v\n", dirPath, err)
				continue
			}

			// Process the parquet files in this directory
			var parquetFiles []hfmodel
			for _, df := range dirFiles {
				if strings.HasSuffix(df.Path, ".parquet") && df.Size > 0 {
					if df.IsLFS {
						df.DownloadLink = fmt.Sprintf(LfsDatasetResolverURL, ModelDatasetName, ModelBranch, df.Path)
					} else {
						df.DownloadLink = fmt.Sprintf(RawDatasetFileURL, ModelDatasetName, ModelBranch, df.Path)
					}
					parquetFiles = append(parquetFiles, df)
				}
			}

			if len(parquetFiles) > 0 {
				if !silentMode {
					fmt.Printf("📦 Processing %d parquet files from %s\n", len(parquetFiles), dirPath)
				}
				processFiles(parquetFiles)
			}
		}
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

// Add a helper function to format file sizes
func formatSize(size int64) string {
	const unit = 1024
	if size < unit {
		return fmt.Sprintf("%d B", size)
	}
	div, exp := int64(unit), 0
	for n := size / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(size)/float64(div), "KMGTPE"[exp])
}

func needsDownload(filePath string, remoteSize int) bool {
	info, err := os.Stat(filePath)
	if os.IsNotExist(err) {
		return true
	}
	return info.Size() != int64(remoteSize)
}

// ***********************************************   All the functions below generated by ChatGPT 3.5, and ChatGPT 4 , with some modifications ***********************************************
func IsValidModelName(modelName string) bool {
	pattern := `^[A-Za-z0-9_\-]+/[A-Za-z0-9\._\-]+$`
	match, _ := regexp.MatchString(pattern, modelName)
	return match
}

func getRedirectLink(url string) (string, error) {

	client := &http.Client{
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if RequiresAuth {
				bearerToken := AuthToken
				req.Header.Add("Authorization", "Bearer "+bearerToken)
			}
			return http.ErrUseLastResponse
		},
	}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", err
	}
	if RequiresAuth {
		// Set the authorization header with the Bearer token
		bearerToken := AuthToken
		req.Header.Add("Authorization", "Bearer "+bearerToken)
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode == 401 && !RequiresAuth {
		return "", fmt.Errorf("\n%s", errorColor("This Repo requires access token, generate an access token form huggingface, and pass it using flag: -t TOKEN"))
	}
	if resp.StatusCode >= 300 && resp.StatusCode <= 399 {
		redirectURL := resp.Header.Get("Location")
		return redirectURL, nil
	}

	return "", fmt.Errorf(errorColor("No redirect found"))
}

func verifyChecksum(filePath, expectedChecksum string) error {
	file, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return err
	}

	actualChecksum := hex.EncodeToString(hasher.Sum(nil))
	if actualChecksum != expectedChecksum {
		return fmt.Errorf("\n%s", errorColor("checksum mismatch: expected ", expectedChecksum, "got ", actualChecksum))
	}

	return nil
}

func downloadChunk(tempFolder string, outputFileName string, idx int, url string, start, end int64, progress chan<- int64) error {
	tmpFileName := path.Join(tempFolder, fmt.Sprintf("%s_%d.tmp", outputFileName, idx))
	var compensationBytes int64 = 12

	// Checking file if exists
	if fi, err := os.Stat(tmpFileName); err == nil { // file exists
		// If file is already completely downloaded
		if fi.Size() == (end - start) {
			// Reflect progress and return
			progress <- fi.Size()
			return nil
		}

		// Fetching size to adjust start byte and compensate for potential corruption
		start = int64(math.Max(float64(start+fi.Size()-compensationBytes), 0.0))

		// Reflecting skipped part in progress, minus compensationBytes so we download them again. Making sure it does not go negative
		progress <- int64(math.Max(float64(fi.Size()-compensationBytes), 0.0))
	}

	client := &http.Client{
		Transport: &http.Transport{},
	}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}

	if RequiresAuth {
		// Set the authorization header with the Bearer token
		bearerToken := AuthToken
		req.Header.Add("Authorization", "Bearer "+bearerToken)
	}

	// Updating the Range header
	rangeHeader := fmt.Sprintf("bytes=%d-%d", start, end-1)
	req.Header.Add("Range", rangeHeader)

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == 401 && !RequiresAuth {
		return fmt.Errorf("\n%s", errorColor("This Repo requires an access token, generate an access token form huggingface, and pass it using flag: -t TOKEN"))
	}

	// Open the file to append/add the new content
	tempFile, err := os.OpenFile(tmpFileName, os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return err
	}
	defer tempFile.Close()

	// Seek to the beginning of the compensation part
	_, err = tempFile.Seek(-compensationBytes, 2) // SEEK_END is 2
	// If seek fails, it probably means the file size is less than compensationBytes
	if err != nil {
		_, err = tempFile.Seek(0, 0) // Seek to start of the file
		if err != nil {
			return err
		}
	}

	buffer := make([]byte, 32768)
	for {
		bytesRead, err := resp.Body.Read(buffer)
		if err != nil && err != io.EOF {
			return err
		}

		if bytesRead == 0 {
			break
		}

		_, err = tempFile.Write(buffer[:bytesRead])
		if err != nil {
			return err
		}

		progress <- int64(bytesRead)
	}

	return nil
}

func mergeFiles(tempFolder, outputFileName string, numChunks int) error {
	outputFile, err := os.Create(outputFileName)
	if err != nil {
		return err
	}
	defer outputFile.Close()

	for i := 0; i < numChunks; i++ {
		tmpFileName := fmt.Sprintf("%s_%d.tmp", path.Base(outputFileName), i)
		tempFileName := path.Join(tempFolder, tmpFileName)
		tempFiles, err := os.ReadDir(tempFolder)
		if err != nil {
			return err
		}
		for _, file := range tempFiles {

			if matched, _ := filepath.Match(tempFileName, path.Join(tempFolder, file.Name())); matched {
				tempFile, err := os.Open(path.Join(tempFolder, file.Name()))
				if err != nil {
					return err
				}
				_, err = io.Copy(outputFile, tempFile)
				if err != nil {
					return err
				}
				err = tempFile.Close()
				if err != nil {
					return err
				}
				err = os.Remove(path.Join(tempFolder, file.Name()))
				if err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func downloadFileMultiThread(tempFolder, url, outputFileName string, silentMode bool, r2cfg *R2Config, skipLocal bool) error {
	// Use larger buffer for network operations
	client := &http.Client{
		Transport: &http.Transport{
			MaxIdleConns:        128,
			MaxIdleConnsPerHost: 128,
			IdleConnTimeout:     90 * time.Second,
			DisableCompression:  true, // Often faster for high-bandwidth connections
			MaxConnsPerHost:     128,
			WriteBufferSize:     64 * 1024, // 64KB
			ReadBufferSize:      64 * 1024, // 64KB
		},
	}
	
	req, err := client.Head(url)
	if err != nil {
		return err
	}
	contentLength, err := strconv.Atoi(req.Header.Get("Content-Length"))
	if err != nil {
		return err
	}

	// Create a pipe for streaming
	pr, pw := io.Pipe()
	var uploadErr error
	var wg sync.WaitGroup

	// Start R2 upload in a goroutine
	if r2cfg != nil && skipLocal {
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer pw.Close()
			
			ctx := context.Background()
			r2Key := strings.TrimPrefix(outputFileName, "/")
			uploadErr = streamMultipartToR2(ctx, *r2cfg, pr, r2Key, int64(contentLength), nil)
		}()
	}

	// Download and write to pipe
	chunkSize := int64(contentLength / NumConnections)
	var downloadWg sync.WaitGroup
	errChan := make(chan error, NumConnections)

	for i := 0; i < NumConnections; i++ {
		start := int64(i) * chunkSize
		end := start + chunkSize
		if i == NumConnections-1 {
			end = int64(contentLength)
		}

		downloadWg.Add(1)
		go func(i int, start, end int64) {
			defer downloadWg.Done()
			
			err := downloadChunkToWriter(url, start, end, pw)
			if err != nil {
				errChan <- fmt.Errorf("chunk %d download failed: %v", i, err)
			}
		}(i, start, end)
	}

	// Wait for downloads to complete
	downloadWg.Wait()
	close(errChan)

	// Check for download errors
	for err := range errChan {
		if err != nil {
			return err
		}
	}

	// Wait for upload to complete
	wg.Wait()
	
	if uploadErr != nil {
		return fmt.Errorf("R2 upload failed: %v", uploadErr)
	}

	return nil
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

func downloadSingleThreaded(url, outputFileName string) error {
	outputFile, err := os.Create(outputFileName)

	if err != nil {
		return err
	}
	defer outputFile.Close()

	// Set the authorization header with the Bearer token

	client := &http.Client{}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err // gracefully handle request err
	}
	if RequiresAuth {
		// Set the authorization header with the Bearer token
		bearerToken := AuthToken
		req.Header.Add("Authorization", "Bearer "+bearerToken)
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}

	defer resp.Body.Close()
	if resp.StatusCode == 401 && !RequiresAuth {
		return fmt.Errorf(errorColor("This Repo requires access token, generate an access token form huggingface, and pass it using flag: -t TOKEN"))

	}
	_, err = io.Copy(outputFile, resp.Body)
	if err != nil {
		return err
	}

	// fmt.Println("\nDownload completed")
	return nil
}

func uploadToR2(ctx context.Context, r2cfg R2Config, localPath string, r2Key string) error {
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
	)
	if err != nil {
		return fmt.Errorf("unable to load SDK config: %v", err)
	}

	client := s3.NewFromConfig(cfg)

	file, err := os.Open(localPath)
	if err != nil {
		return fmt.Errorf("unable to open file: %v", err)
	}
	defer file.Close()

	_, err = client.PutObject(ctx, &s3.PutObjectInput{
		Bucket: aws.String(r2cfg.BucketName),
		Key:    aws.String(r2Key),
		Body:   file,
	})
	
	return err
}

func streamSmallFileToR2(url, outputFileName string, r2cfg R2Config) error {
	// Implement the logic to stream a small file directly to R2
	// This is a placeholder and should be replaced with the actual implementation
	return fmt.Errorf("streaming small file to R2 is not implemented")
}

// Add new function for streaming chunks
func downloadChunkToWriter(url string, start, end int64, writer io.Writer) error {
	client := &http.Client{}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	
	if RequiresAuth {
		req.Header.Add("Authorization", "Bearer "+AuthToken)
	}
	
	rangeHeader := fmt.Sprintf("bytes=%d-%d", start, end-1)
	req.Header.Add("Range", rangeHeader)

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	buffer := make([]byte, 32*1024)
	_, err = io.CopyBuffer(writer, resp.Body, buffer)
	return err
}

// Add new function for streaming to R2
func streamToR2(ctx context.Context, r2cfg R2Config, reader io.Reader, key string, contentLength int) error {
	client := createR2Client(ctx, r2cfg)
	length := int64(contentLength)
	_, err := client.PutObject(ctx, &s3.PutObjectInput{
		Bucket:        aws.String(r2cfg.BucketName),
		Key:           aws.String(key),
		Body:          reader,
		ContentLength: &length,
	})

	return err
}

// Helper function for simple uploads
func streamSimpleToR2(ctx context.Context, r2cfg R2Config, reader io.Reader, key string, contentLength int64, progress *uploadProgress) error {
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
	
	return err
}

func min(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

