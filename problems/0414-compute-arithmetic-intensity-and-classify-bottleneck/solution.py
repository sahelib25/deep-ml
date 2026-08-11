def compute_arithmetic_intensity(flops: float, bytes_accessed: float, peak_performance: float, peak_bandwidth: float) -> dict:
    """
    Analyze a computational kernel using the Roofline Model.
    
    Args:
        flops: Total floating-point operations of the kernel
        bytes_accessed: Total bytes transferred to/from memory
        peak_performance: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)
    
    Returns:
        Dictionary with arithmetic_intensity, ridge_point, bottleneck,
        achieved_performance, and utilization_percent
    """
    ai = flops / bytes_accessed
    rp = peak_performance/peak_bandwidth
    bottleneck = ('compute-bound' if ai >= rp else 'memory-bound')
    ap = min(peak_performance, ai * peak_bandwidth)
    up = (ap / peak_performance) * 100
    ai = {
        'arithmetic_intensity': ai,
        'ridge_point': rp,
        'bottleneck': bottleneck,
        'achieved_performance': ap,
        'utilization_percent': up
    }
    return ai