def classify_llm_phases(num_params: int, sequence_length: int, batch_size: int, bytes_per_param: int, peak_flops: float, peak_bandwidth: float) -> dict:
    """
    Analyze prefill and decode phases of LLM inference using the Roofline Model.

    Args:
        num_params: Total number of model parameters
        sequence_length: Number of input tokens processed during prefill
        batch_size: Number of sequences processed in parallel during decode
        bytes_per_param: Memory footprint per parameter (e.g., 2 for FP16)
        peak_flops: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)

    Returns:
        Dictionary containing ridge_point and analysis dicts for 'prefill' and 'decode',
        each with total_flops, memory_bytes, arithmetic_intensity, bottleneck,
        achieved_flops, and utilization_percent.
    """
    ridge_point = peak_flops / peak_bandwidth
    prefill_decode = {
        'ridge_point': ridge_point,
        'prefill': {},
        'decode': {}
    }

    prefill_decode['prefill']['total_flops'] =  2 * num_params * sequence_length 
    prefill_decode['prefill']['memory_bytes'] = num_params * bytes_per_param
    prefill_decode['prefill']['arithmetic_intensity'] = prefill_decode['prefill']['total_flops']  / prefill_decode['prefill']['memory_bytes']
    prefill_decode['prefill']['bottleneck'] = ('compute-bound' if prefill_decode['prefill']['arithmetic_intensity'] >= ridge_point else 'memory-bound')
    prefill_decode['prefill']['achieved_flops'] = min(
        peak_flops,
        prefill_decode["prefill"]["arithmetic_intensity"] * peak_bandwidth
    )
    prefill_decode["prefill"]["utilization_percent"] = (
        prefill_decode["prefill"]["achieved_flops"]
        / peak_flops
        * 100
    )
    
        # -------------------------
    # Decode
    # -------------------------

    prefill_decode["decode"]["total_flops"] = (
        2 * num_params * batch_size
    )

    prefill_decode["decode"]["memory_bytes"] = (
        num_params * bytes_per_param
    )

    prefill_decode["decode"]["arithmetic_intensity"] = (
        prefill_decode["decode"]["total_flops"]
        / prefill_decode["decode"]["memory_bytes"]
    )

    prefill_decode["decode"]["bottleneck"] = (
        "compute-bound"
        if prefill_decode["decode"]["arithmetic_intensity"] >= ridge_point
        else "memory-bound"
    )

    prefill_decode["decode"]["achieved_flops"] = min(
        peak_flops,
        prefill_decode["decode"]["arithmetic_intensity"] * peak_bandwidth
    )

    prefill_decode["decode"]["utilization_percent"] = (
        prefill_decode["decode"]["achieved_flops"]
        / peak_flops
        * 100
    )
    return prefill_decode