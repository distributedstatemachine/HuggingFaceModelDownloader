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
	Subfolder       string // Custom subfolder (e.g., "mlfoundations-dclm-baseline-1.0-parquet")
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

func DownloadModel(ModelDatasetName string, AppendFilterToPath bool, SkipSHA bool, IsDataset bool, DestinationBasePath string, ModelBranch string, concurrentConnections int, token string, silentMode bool, r2cfg *R2Config, skipLocal bool) error {
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
				// Handle path prefix differently for DCLM dataset
				var r2Key string
				if strings.Contains(ModelDatasetName, "mlfoundations/dclm-baseline-1.0-parquet") {
					// Trim away the long prefix path and extract just the global/local shard part
					trimmedPath := file.Path
					if strings.Contains(trimmedPath, "global-shard_") {
						// Extract the pattern global-shard_XX_of_10/local-shard_Y_of_10/filename.parquet
						index := strings.Index(trimmedPath, "global-shard_")
						if index > 0 {
							trimmedPath = trimmedPath[index:]
						}
					}
					r2Key = fmt.Sprintf("%s/%s", r2cfg.Subfolder, trimmedPath)
				} else {
					r2Key = fmt.Sprintf("%s/%s", r2cfg.Subfolder, strings.TrimPrefix(file.Path, "data/"))
				}

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
				// Handle path prefix differently for DCLM dataset
				var r2Key string
				if strings.Contains(ModelDatasetName, "mlfoundations/dclm-baseline-1.0-parquet") {
					// Trim away the long prefix path and extract just the global/local shard part
					trimmedPath := file.Path
					if strings.Contains(trimmedPath, "global-shard_") {
						// Extract the pattern global-shard_XX_of_10/local-shard_Y_of_10/filename.parquet
						index := strings.Index(trimmedPath, "global-shard_")
						if index > 0 {
							trimmedPath = trimmedPath[index:]
						}
					}
					r2Key = fmt.Sprintf("%s/%s", r2cfg.Subfolder, trimmedPath)
				} else {
					r2Key = fmt.Sprintf("%s/%s", r2cfg.Subfolder, strings.TrimPrefix(file.Path, "data/"))
				}

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
	err = processHFFolderTree(modelPath, IsDataset, SkipSHA, ModelDatasetName, ModelBranch, "", silentMode, r2cfg, skipLocal, processFiles)
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

	// Check if we're processing the dclm dataset
	isDCLMDataset := false
	if ModelDatasetName == "mlfoundations/dclm-baseline-1.0-parquet" {
		isDCLMDataset = true
	}

	// Build the correct API URL
	var url string
	if IsDataset {
		if folderName == "" {
			// For DCLM dataset, use filtered root path instead of data
			if isDCLMDataset {
				url = fmt.Sprintf(JsonDatasetFileTreeURL, ModelDatasetName, ModelBranch, "filtered")
			} else {
				url = fmt.Sprintf(JsonDatasetFileTreeURL, ModelDatasetName, ModelBranch, "data")
			}
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

	// Special handling for DCLM dataset
	if isDCLMDataset {
		if folderName == "" {
			// First level - navigate to the filtered directory
			for _, file := range files {
				if file.IsDirectory || strings.Contains(file.Path, "filtered") {
					newPath := file.Path
					if !silentMode {
						fmt.Printf("📁 Entering directory: %s\n", newPath)
					}
					// Recursively process this directory
					err := processHFFolderTree(modelPath, IsDataset, SkipSHA, ModelDatasetName, ModelBranch, newPath, silentMode, r2cfg, skipLocal, processFiles)
					if err != nil {
						fmt.Printf("⚠️ Error processing directory %s: %v\n", newPath, err)
					}
				}
			}
		} else if strings.Contains(folderName, "processed_data") {
			// We're in the processed_data directory, process all global shards
			for globalIdx := 0; globalIdx <= 10; globalIdx++ {
				globalShardPath := fmt.Sprintf("%s/global-shard_%02d_of_10", folderName, globalIdx)
				if !silentMode {
					fmt.Printf("📁 Processing global shard: %s\n", globalShardPath)
				}

				// Fetch contents of this global shard
				globalShardUrl := fmt.Sprintf(JsonDatasetFileTreeURL, ModelDatasetName, ModelBranch, globalShardPath)
				globalShardFiles, err := fetchFileList(globalShardUrl)
				if err != nil {
					fmt.Printf("⚠️ Error fetching global shard %s: %v\n", globalShardPath, err)
					continue
				}

				// Process all local shards within each global shard
				var parquetFiles []hfmodel
				for _, gf := range globalShardFiles {
					// Check if it's a local shard directory
					if gf.IsDirectory || strings.Contains(gf.Path, "local-shard") {
						localShardPath := gf.Path

						// Fetch contents of this local shard
						localShardUrl := fmt.Sprintf(JsonDatasetFileTreeURL, ModelDatasetName, ModelBranch, localShardPath)
						localShardFiles, err := fetchFileList(localShardUrl)
						if err != nil {
							fmt.Printf("⚠️ Error fetching local shard %s: %v\n", localShardPath, err)
							continue
						}

						// Add all parquet files to our list
						for _, lf := range localShardFiles {
							if strings.HasSuffix(lf.Path, ".parquet") && lf.Size > 0 {
								if lf.IsLFS {
									lf.DownloadLink = fmt.Sprintf(LfsDatasetResolverURL, ModelDatasetName, ModelBranch, lf.Path)
								} else {
									lf.DownloadLink = fmt.Sprintf(RawDatasetFileURL, ModelDatasetName, ModelBranch, lf.Path)
								}
								parquetFiles = append(parquetFiles, lf)
							}
						}
					} else if strings.HasSuffix(gf.Path, ".parquet") && gf.Size > 0 {
						// In case parquet files are directly in the global shard
						if gf.IsLFS {
							gf.DownloadLink = fmt.Sprintf(LfsDatasetResolverURL, ModelDatasetName, ModelBranch, gf.Path)
						} else {
							gf.DownloadLink = fmt.Sprintf(RawDatasetFileURL, ModelDatasetName, ModelBranch, gf.Path)
						}
						parquetFiles = append(parquetFiles, gf)
					}
				}

				if len(parquetFiles) > 0 {
					if !silentMode {
						fmt.Printf("📦 Processing %d parquet files from global shard %02d\n", len(parquetFiles), globalIdx)
					}
					processFiles(parquetFiles)
				}
			}
		} else {
			// We're in an intermediate directory, keep going deeper
			for _, file := range files {
				if file.IsDirectory || !strings.HasSuffix(file.Path, ".parquet") {
					newPath := file.Path
					if !silentMode {
						fmt.Printf("📁 Entering directory: %s\n", newPath)
					}
					// Recursively process this directory
					err := processHFFolderTree(modelPath, IsDataset, SkipSHA, ModelDatasetName, ModelBranch, newPath, silentMode, r2cfg, skipLocal, processFiles)
					if err != nil {
						fmt.Printf("⚠️ Error processing directory %s: %v\n", newPath, err)
					}
				}
			}
		}
		return nil
	}

	// Standard handling for other datasets (unchanged)
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

func min(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
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
