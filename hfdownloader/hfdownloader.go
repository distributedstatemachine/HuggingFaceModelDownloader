package hfdownloader

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
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

// custom httpClient to use our custom DNS resolver.
var httpClient *http.Client

func init() {
	// Initialize random seed for jitter calculations
	rand.Seed(time.Now().UnixNano())

	// To solve DNS timeout issues, and resolve faster, we use  cloudflare's DNS
	r := &net.Resolver{
		PreferGo: true,
		Dial: func(ctx context.Context, network, address string) (net.Conn, error) {
			d := &net.Dialer{Timeout: 5 * time.Second}
			return d.DialContext(ctx, network, "1.1.1.1:53")
		},
	}

	dialer := &net.Dialer{
		Timeout:   10 * time.Second,
		KeepAlive: 30 * time.Second,
		Resolver:  r,
	}

	transport := &http.Transport{
		DialContext:         dialer.DialContext,
		TLSHandshakeTimeout: 10 * time.Second,
		MaxIdleConns:        NumConnections,
		MaxIdleConnsPerHost: NumConnections,
		IdleConnTimeout:     30 * time.Second,
		DisableKeepAlives:   false,
	}

	// Set a longer timeout for the HTTP client (10 minutes)
	// Individual requests will use context with their own timeouts
	httpClient = &http.Client{
		Transport: transport,
		Timeout:   10 * time.Minute,
	}
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
	files map[string]int64 // map of file key to file size
	mu    sync.RWMutex
}

// Add this function to pre-fetch existing files
func buildR2Cache(ctx context.Context, r2cfg *R2Config, prefix string) (*R2FileCache, error) {
	client := createR2Client(ctx, *r2cfg)
	cache := &R2FileCache{
		files: make(map[string]int64),
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
			cache.files[*obj.Key] = *obj.Size
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

// Add method to check if file exists and has the expected size
func (c *R2FileCache) ExistsWithSize(key string, expectedSize int64) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	size, exists := c.files[key]
	return exists && size == expectedSize
}

// Get the file size
func (c *R2FileCache) GetSize(key string) (int64, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	size, exists := c.files[key]
	return size, exists
}

// DownloadState represents the current state of a model download
type DownloadState struct {
	ModelName      string          `json:"model_name"`
	Branch         string          `json:"branch"`
	TotalFiles     int             `json:"total_files"`
	CompletedFiles map[string]bool `json:"completed_files"`
	LastUpdate     time.Time       `json:"last_update"`
	StartTime      time.Time       `json:"start_time"`
}

// saveDownloadState saves the current download state to a file
func saveDownloadState(state *DownloadState, modelName string) error {
	// Create state directory if it doesn't exist
	stateDir := filepath.Join(os.TempDir(), "hfdownloader-state")
	if err := os.MkdirAll(stateDir, 0755); err != nil {
		return fmt.Errorf("failed to create state directory: %v", err)
	}

	// Create filename from sanitized model name
	safeModelName := strings.ReplaceAll(modelName, "/", "_")
	stateFile := filepath.Join(stateDir, fmt.Sprintf("%s.json", safeModelName))

	// Update timestamp
	state.LastUpdate = time.Now()

	// Write state to file
	file, err := os.Create(stateFile)
	if err != nil {
		return fmt.Errorf("failed to create state file: %v", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(state); err != nil {
		return fmt.Errorf("failed to encode state: %v", err)
	}

	return nil
}

// loadDownloadState loads the download state from a file
func loadDownloadState(modelName string) (*DownloadState, error) {
	// Create filename from sanitized model name
	safeModelName := strings.ReplaceAll(modelName, "/", "_")
	stateDir := filepath.Join(os.TempDir(), "hfdownloader-state")
	stateFile := filepath.Join(stateDir, fmt.Sprintf("%s.json", safeModelName))

	// Check if state file exists
	if _, err := os.Stat(stateFile); os.IsNotExist(err) {
		return nil, nil // No state file exists
	}

	// Read state file
	file, err := os.Open(stateFile)
	if err != nil {
		return nil, fmt.Errorf("failed to open state file: %v", err)
	}
	defer file.Close()

	// Decode state
	state := &DownloadState{
		CompletedFiles: make(map[string]bool),
	}
	if err := json.NewDecoder(file).Decode(state); err != nil {
		return nil, fmt.Errorf("failed to decode state: %v", err)
	}

	// Check if state is stale (older than 7 days)
	if time.Since(state.LastUpdate) > 7*24*time.Hour {
		os.Remove(stateFile) // Remove stale state file
		return nil, nil
	}

	return state, nil
}

func DownloadModel(ModelDatasetName string, AppendFilterToPath bool, SkipSHA bool, IsDataset bool, DestinationBasePath string, ModelBranch string, concurrentConnections int, token string, silentMode bool, r2cfg *R2Config, skipLocal bool, hfPrefix string) error {
	// Create a cancellable context with a 24-hour timeout
	ctx, cancel := context.WithTimeout(context.Background(), 24*time.Hour)
	defer cancel()

	// Load existing download state
	downloadState, err := loadDownloadState(ModelDatasetName)
	if err != nil {
		fmt.Printf("Warning: Failed to load download state: %v\n", err)
	}

	// Initialize new state if needed
	if downloadState == nil {
		downloadState = &DownloadState{
			ModelName:      ModelDatasetName,
			Branch:         ModelBranch,
			TotalFiles:     0,
			CompletedFiles: make(map[string]bool),
			StartTime:      time.Now(),
			LastUpdate:     time.Now(),
		}
		fmt.Println("🆕 Starting new download session")
	} else {
		fmt.Printf("🔄 Resuming download from previous session (started %s)\n",
			time.Since(downloadState.StartTime).Round(time.Minute))
		fmt.Printf("💾 Previously completed: %d/%d files\n",
			len(downloadState.CompletedFiles), downloadState.TotalFiles)
	}

	// Build cache of existing files
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
			// Add panic recovery to prevent worker crashes from bringing down the entire process
			defer func() {
				if r := recover(); r != nil {
					stack := make([]byte, 8192)
					length := runtime.Stack(stack, false)
					errMsg := fmt.Sprintf("❌ Worker %d panicked: %v\n%s", workerID, r, stack[:length])
					fmt.Println(errMsg)
					results <- fmt.Errorf("worker %d panicked: %v", workerID, r)
				}
			}()
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

				// Check if file exists with correct size using ExistsWithSize
				if cache.ExistsWithSize(r2Key, int64(file.Size)) {
					if !silentMode {
						fmt.Printf("Skipping %s - already exists in R2 with correct size\n", r2Key)
					}
					completedFiles.Add(1)
					continue
				} else if existingSize, exists := cache.GetSize(r2Key); exists {
					// File exists but with incorrect size, delete it and reupload
					fmt.Printf("File %s exists with incorrect size (expected: %s, actual: %s). Deleting and reuploading...\n",
						r2Key, formatSize(int64(file.Size)), formatSize(existingSize))

					client := createR2Client(ctx, *r2cfg)
					_, deleteErr := client.DeleteObject(ctx, &s3.DeleteObjectInput{
						Bucket: aws.String(r2cfg.BucketName),
						Key:    aws.String(r2Key),
					})
					if deleteErr != nil {
						fmt.Printf("Warning: Failed to delete incomplete file %s: %v\n", r2Key, deleteErr)
					}
				}

				downloadURL := fmt.Sprintf("https://huggingface.co/datasets/%s/resolve/%s/%s",
					ModelDatasetName,
					ModelBranch,
					file.Path,
				)

				fmt.Printf("Worker %d: Starting download of %s\n", workerID, file.Path)

				// Create download-specific context with longer timeout for large files (30 minutes)
				downloadCtx, cancelDownload := context.WithTimeout(ctx, 30*time.Minute)
				defer cancelDownload()

				// Create request with context
				req, err := http.NewRequestWithContext(downloadCtx, "GET", downloadURL, nil)
				if err != nil {
					fmt.Printf("Error creating request for %s: %v\n", file.Path, err)
					results <- fmt.Errorf("failed to create request for %s: %v", file.Path, err)
					continue
				}

				if RequiresAuth {
					req.Header.Add("Authorization", "Bearer "+AuthToken)
				}
				req.Header.Add("User-Agent", "Mozilla/5.0")

				// Download file with retry logic
				var resp *http.Response
				downloadErr := retryWithBackoff(func() error {
					var err error
					resp, err = httpClient.Do(req)
					if err != nil {
						return fmt.Errorf("request failed: %v", err)
					}

					if resp.StatusCode != http.StatusOK {
						bodyBytes, _ := io.ReadAll(resp.Body)
						resp.Body.Close()
						return fmt.Errorf("bad status: %d, body: %s", resp.StatusCode, string(bodyBytes))
					}

					return nil
				}, 5, 1*time.Second, 30*time.Second)

				if downloadErr != nil {
					if resp != nil && resp.Body != nil {
						resp.Body.Close()
					}
					fmt.Printf("Error downloading %s after retries: %v\n", file.Path, downloadErr)
					results <- fmt.Errorf("failed to download %s: %v", file.Path, downloadErr)
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

				// Mark as completed in download state
				downloadState.CompletedFiles[file.Path] = true
				// Save download state periodically (every ~5 files)
				if completedFiles.Load()%5 == 0 {
					if err := saveDownloadState(downloadState, ModelDatasetName); err != nil {
						fmt.Printf("Warning: Failed to save download state: %v\n", err)
					}
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

		// Update total file count in download state
		// Should only count files that need downloading
		fileCount := 0
		for _, file := range files {
			if !file.IsDirectory && !file.FilterSkip && file.Size > 0 {
				fileCount++
			}
		}

		if downloadState.TotalFiles == 0 {
			downloadState.TotalFiles = fileCount
		} else {
			downloadState.TotalFiles += fileCount
		}

		// Save state
		if err := saveDownloadState(downloadState, ModelDatasetName); err != nil {
			fmt.Printf("Warning: Failed to save download state: %v\n", err)
		}

		// First, filter files that need to be processed
		for _, file := range files {
			if !file.IsDirectory && !file.FilterSkip && file.Size > 0 {
				r2Key := fmt.Sprintf("%s/%s", r2cfg.Subfolder, strings.TrimPrefix(file.Path, fmt.Sprintf("%s/", hfPrefix)))

				totalSize += int64(file.Size)

				// Check if file is already in completed files list
				if downloadState.CompletedFiles[file.Path] {
					fmt.Printf("Skipping %s - marked as completed in saved state\n", file.Path)
					skippedSize += int64(file.Size)
					skippedCount++
					continue
				}

				if cache.ExistsWithSize(r2Key, int64(file.Size)) {
					// File exists in R2 with correct size - mark as completed
					downloadState.CompletedFiles[file.Path] = true
					skippedSize += int64(file.Size)
					skippedCount++
					continue
				} else if existingSize, exists := cache.GetSize(r2Key); exists {
					// File exists but with incorrect size, will be reuploaded
					fmt.Printf("File %s exists with incorrect size (expected: %s, actual: %s). Will be deleted and reuploaded.\n",
						r2Key, formatSize(int64(file.Size)), formatSize(existingSize))
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

	// Start watchdog to monitor progress
	stopWatchdog := make(chan struct{})
	go func() {
		ticker := time.NewTicker(2 * time.Minute) // Check progress every 2 minutes
		defer ticker.Stop()

		var lastCompleted int32
		staleCount := 0

		for {
			select {
			case <-ticker.C:
				currentCompleted := completedFiles.Load()

				if currentCompleted == lastCompleted && lastCompleted > 0 {
					staleCount++
					// Longer stale detection for large files (10 minutes = 5 checks)
					fmt.Printf("⚠️ Warning: No progress detected for %d minutes\n", staleCount*2)

					if staleCount >= 15 { // No progress for 30 minutes
						fmt.Println("🔄 Progress appears to be stalled for too long!")
						// We'll log this but not force cancel as it could be a very large file
						staleCount = 0 // Reset to avoid multiple warnings
					}
				} else {
					if lastCompleted > 0 {
						fmt.Printf("📊 Progress update: %d files completed (+%d new)\n",
							currentCompleted, currentCompleted-lastCompleted)
					}
					staleCount = 0
					lastCompleted = currentCompleted
				}
			case <-stopWatchdog:
				fmt.Println("🔍 Watchdog stopped - download completed or canceled")
				return
			}
		}
	}()

	// Start processing
	err = processHFFolderTree(modelPath, IsDataset, SkipSHA, ModelDatasetName, ModelBranch, "", silentMode, r2cfg, skipLocal, processFiles, hfPrefix)
	if err != nil {
		close(stopWatchdog)
		return fmt.Errorf("error processing file tree: %v", err)
	}

	// Stop watchdog
	close(stopWatchdog)

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
		// Save state before returning error
		if err := saveDownloadState(downloadState, ModelDatasetName); err != nil {
			fmt.Printf("Warning: Failed to save download state: %v\n", err)
		}
		return fmt.Errorf("encountered errors: %v", errors)
	}

	// Save final state
	fmt.Println("💾 Saving final download state")
	if err := saveDownloadState(downloadState, ModelDatasetName); err != nil {
		fmt.Printf("Warning: Failed to save final download state: %v\n", err)
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
	// Create a context with timeout for the API request (2 minutes should be plenty)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	if RequiresAuth {
		req.Header.Add("Authorization", "Bearer "+AuthToken)
	}
	req.Header.Add("User-Agent", "Mozilla/5.0")

	var resp *http.Response
	var files []hfmodel

	// Use retry with backoff for API requests
	fetchErr := retryWithBackoff(func() error {
		var err error
		resp, err = httpClient.Do(req)
		if err != nil {
			return fmt.Errorf("request failed: %v", err)
		}

		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			return fmt.Errorf("bad status: %d, body: %s", resp.StatusCode, string(bodyBytes))
		}

		// Decode response
		files = []hfmodel{}
		if err := json.NewDecoder(resp.Body).Decode(&files); err != nil {
			resp.Body.Close()
			return fmt.Errorf("failed to decode response: %v", err)
		}

		resp.Body.Close()
		return nil
	}, 5, 1*time.Second, 10*time.Second)

	if fetchErr != nil {
		return nil, fmt.Errorf("failed to fetch file list after retries: %v", fetchErr)
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

	// Check for existing multipart uploads that we might resume
	var uploadID string
	listResp, err := client.ListMultipartUploads(ctx, &s3.ListMultipartUploadsInput{
		Bucket: aws.String(r2cfg.BucketName),
		Prefix: aws.String(key),
	})

	if err == nil && listResp.Uploads != nil {
		for _, upload := range listResp.Uploads {
			if *upload.Key == key {
				// Found an existing upload for this exact key - let's try to resume it
				uploadID = *upload.UploadId
				fmt.Printf("Found existing multipart upload for %s (ID: %s) - attempting to resume\n", key, uploadID)

				// Get existing parts to potentially resume from
				listPartsResp, listErr := client.ListParts(ctx, &s3.ListPartsInput{
					Bucket:   aws.String(r2cfg.BucketName),
					Key:      aws.String(key),
					UploadId: aws.String(uploadID),
				})

				if listErr == nil && len(listPartsResp.Parts) > 0 {
					fmt.Printf("Found %d previously uploaded parts for %s\n", len(listPartsResp.Parts), key)
					// TODO: In a more complex implementation, we could resume from these parts
					// Currently, we'll just abort and start fresh to ensure consistency
				}

				break
			}
		}

		// If we didn't find a matching upload or decide not to resume, abort all existing uploads
		if uploadID == "" {
			for _, upload := range listResp.Uploads {
				_, abortErr := client.AbortMultipartUpload(ctx, &s3.AbortMultipartUploadInput{
					Bucket:   aws.String(r2cfg.BucketName),
					Key:      upload.Key,
					UploadId: upload.UploadId,
				})
				if abortErr != nil {
					fmt.Printf("Warning: Failed to abort incomplete upload for %s: %v\n", *upload.Key, abortErr)
				}
			}
		}
	}

	// Create new multipart upload if we don't have one to resume
	if uploadID == "" {
		resp, err := client.CreateMultipartUpload(ctx, &s3.CreateMultipartUploadInput{
			Bucket: aws.String(r2cfg.BucketName),
			Key:    aws.String(key),
		})
		if err != nil {
			return fmt.Errorf("failed to create multipart upload: %v", err)
		}
		uploadID = *resp.UploadId
		fmt.Printf("Created new multipart upload for %s (ID: %s)\n", key, uploadID)
	}

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
// Helper function to determine if an error is transient and retryable
func isTransientError(err error) bool {
	if err == nil {
		return false
	}

	// Check for network timeouts and temporary failures
	if netErr, ok := err.(net.Error); ok && (netErr.Timeout() || netErr.Temporary()) {
		return true
	}

	// Check for HTTP retryable status codes
	errStr := err.Error()
	if strings.Contains(errStr, "status 429") || // Too Many Requests
		strings.Contains(errStr, "status 500") || // Internal Server Error
		strings.Contains(errStr, "status 502") || // Bad Gateway
		strings.Contains(errStr, "status 503") || // Service Unavailable
		strings.Contains(errStr, "status 504") { // Gateway Timeout
		return true
	}

	// Check for common AWS S3/R2 retryable errors
	if strings.Contains(errStr, "RequestTimeout") ||
		strings.Contains(errStr, "SlowDown") ||
		strings.Contains(errStr, "InternalError") ||
		strings.Contains(errStr, "connection reset") ||
		strings.Contains(errStr, "EOF") ||
		strings.Contains(errStr, "broken pipe") {
		return true
	}

	return false
}

// Retry an operation with exponential backoff
func retryWithBackoff(operation func() error, maxRetries int, initialBackoff, maxBackoff time.Duration) error {
	var err error

	for attempt := 0; attempt < maxRetries; attempt++ {
		err = operation()
		if err == nil {
			return nil
		}

		if !isTransientError(err) {
			return fmt.Errorf("permanent error (not retrying): %v", err)
		}

		if attempt == maxRetries-1 {
			break // Last attempt failed, exit loop
		}

		// Calculate backoff with jitter
		backoff := time.Duration(float64(initialBackoff) * math.Pow(2, float64(attempt)))
		if backoff > maxBackoff {
			backoff = maxBackoff
		}
		// Add jitter (±20%)
		jitter := time.Duration(float64(backoff) * (0.8 + 0.4*rand.Float64()))

		fmt.Printf("Retrying operation after %v (attempt %d/%d): %v\n",
			jitter.Round(time.Millisecond), attempt+1, maxRetries, err)
		time.Sleep(jitter)
	}

	return fmt.Errorf("operation failed after %d retries: %v", maxRetries, err)
}

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
