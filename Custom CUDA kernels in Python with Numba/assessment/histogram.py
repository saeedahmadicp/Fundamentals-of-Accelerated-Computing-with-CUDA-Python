# Add your solution here
@cuda.jit
def cuda_histogram(x, xmin, xmax, histogram_out):
    '''Increment bin counts in histogram_out, given histogram range [xmin, xmax).'''
    
    ##start of the current thread
    start = cuda.grid(1)
    
    ## stride == total number of threads 
    stride = cuda.gridsize(1)
    
    
    ##implementing the rest of the logic for bins
    nbins = histogram_out.shape[0]
    bin_width = (xmax - xmin) / nbins
    
    ## now, we will process bins using different threads 
    for idx in range(start, len(x), stride):
        element = x[idx]
        bin_number = np.int32((element - xmin)/bin_width)
        if bin_number >= 0 and bin_number < histogram_out.shape[0]:
            # only increment if in range
            cuda.atomic.add(histogram_out, bin_number, 1)