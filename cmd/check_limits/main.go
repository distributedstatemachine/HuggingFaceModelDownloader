package main

import (
	"fmt"
	"runtime"
	 "time"
)

const (
	// Fixed buffer sizes (in bytes)
	_256MB int64 = 256 * 1024 * 1024
	_128MB int64 = 128 * 1024 * 1024
	_5MB   int64 = 5 * 1024 * 1024
	_5GB   int64 = 5 * 1024 * 1024 * 1024

	// Retry configuration
	maxRetries = 3
	retryDelay = 500 * time.Millisecond
)

// Configuration variables computed at runtime
var (
	// System resources
	systemMemory = uint64(runtime.GOMAXPROCS(-1)) * 1024 * 1024 * 1024

	// Buffer sizes
	streamBufferSize = min(_256MB, int64(systemMemory/4))  // 256MB or 25% of memory
	bufferSize       = min(_128MB, int64(systemMemory/8))  // 128MB or 12.5% of memory

	// Concurrent operations
	maxConcurrent   = max(2, runtime.NumCPU()/2) // Half of CPU cores, minimum 2
	maxPartsPerFile = max(4, runtime.NumCPU())   // At least 4, scales with CPUs

	// R2/S3 limits
	multipartThreshold int64 = _5MB  // 5MB (S3 minimum)
	maxPartSize       int64 = _5GB   // 5GB (S3 maximum)
	chunkSize         int64 = min(_256MB, _5GB) // 256MB chunks, respect S3 limits
)

func min(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

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

func calculateMaxSafeValues() {
	// Get system info
	cpuCores := runtime.NumCPU()
	memTotal := uint64(runtime.GOMAXPROCS(-1)) * 1024 * 1024 * 1024 // Approx total memory

	// Calculate safe maximums
	maxSafeWorkers := cpuCores * 2                    // 2x CPU cores
	maxSafeMemPerWorker := int64(memTotal) / 4        // 25% of total memory
	maxSafeBufferSize := maxSafeMemPerWorker / int64(maxSafeWorkers)
	maxSafePartsPerFile := min(int64(10000), int64(cpuCores*4))     // S3 limit is 10,000 parts
	maxSafeChunkSize := min(_5GB, maxSafeBufferSize)  // Cannot exceed S3's 5GB limit

	fmt.Println("\n=== System Resources ===")
	fmt.Printf("CPU Cores: %d\n", cpuCores)
	fmt.Printf("Total Memory: %s\n", formatSize(int64(memTotal)))

	fmt.Println("\n=== Maximum Safe Values ===")
	fmt.Printf("Max Safe Workers: %d\n", maxSafeWorkers)
	fmt.Printf("Max Safe Memory Per Worker: %s\n", formatSize(maxSafeMemPerWorker))
	fmt.Printf("Max Safe Buffer Size: %s\n", formatSize(maxSafeBufferSize))
	fmt.Printf("Max Safe Parts Per File: %d\n", maxSafePartsPerFile)
	fmt.Printf("Max Safe Chunk Size: %s\n", formatSize(maxSafeChunkSize))

	fmt.Println("\n=== Recommended Configuration ===")
	fmt.Printf("const (\n")
	fmt.Printf("    streamBufferSize    = %d  // %s\n", maxSafeBufferSize, formatSize(maxSafeBufferSize))
	fmt.Printf("    multipartThreshold  = %d  // %s\n", _5MB, formatSize(_5MB))
	fmt.Printf("    chunkSize          = %d  // %s\n", maxSafeChunkSize, formatSize(maxSafeChunkSize))
	fmt.Printf("    maxConcurrent      = %d   // workers\n", maxSafeWorkers/2) // Conservative
	fmt.Printf("    maxPartsPerFile    = %d   // parts\n", maxSafePartsPerFile)
	fmt.Printf("    bufferSize         = %d  // %s\n", maxSafeBufferSize/2, formatSize(maxSafeBufferSize/2))
	fmt.Printf(")\n")

	fmt.Println("\n=== Performance Estimates ===")
	theoreticalMaxThroughput := float64(maxSafeWorkers) * float64(maxSafeChunkSize) / 1024 / 1024 // MB/s
	fmt.Printf("Theoretical Max Throughput: %.2f MB/s\n", theoreticalMaxThroughput)
	fmt.Printf("Estimated Time for 8TB: %.1f hours\n", (8*1024*1024)/theoreticalMaxThroughput/3600)

	fmt.Println("\n=== S3/R2 Limits ===")
	fmt.Printf("Minimum Part Size: %s\n", formatSize(_5MB))
	fmt.Printf("Maximum Part Size: %s\n", formatSize(_5GB))
	fmt.Printf("Maximum Parts Per Upload: 10,000\n")
	fmt.Printf("Maximum Object Size: %s\n", formatSize(_5GB*10000))
}

func main() {
	fmt.Println("Calculating Maximum Safe Configuration Values...")
	calculateMaxSafeValues()
}