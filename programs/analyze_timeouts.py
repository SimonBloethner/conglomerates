#!/usr/bin/env python3
"""
Analyze timeout patterns from parallel counterfactual runs
"""

import os
import re
import glob
from collections import defaultdict

def parse_log_file(filename):
    """Parse a single SLURM log file and extract timeout/completion info"""
    results = []
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
            
            # Extract timeout information
            timeout_matches = re.findall(r'TIMEOUT: share=([\d.]+), exp=(\d+), seed=(\d+), runtime=([\d.]+)s', content)
            for share, exp, seed, runtime in timeout_matches:
                # Try to find the last step reached
                progress_matches = re.findall(r'PROGRESS: Step (\d+)/\d+', content)
                last_step = int(progress_matches[-1]) if progress_matches else 0
                
                # Try to find performance characteristics at timeout
                perf_matches = re.findall(r'PERF: Step (\d+): (\d+) conglomerates, max_size=(\d+), avg_size=([\d.]+), total_members=(\d+), solo=(\d+), wealth_range=\[([\d.e+-]+), ([\d.e+-]+)\]', content)
                last_perf = perf_matches[-1] if perf_matches else None
                
                results.append({
                    'type': 'TIMEOUT',
                    'share': float(share),
                    'exp': int(exp),
                    'seed': int(seed),
                    'runtime': float(runtime),
                    'last_step': last_step,
                    'last_perf': last_perf,
                    'file': filename
                })
            
            # Extract successful completions
            success_matches = re.findall(r'Completed: share=([\d.]+), exp=(\d+), runtime=([\d.]+)s', content)
            for share, exp, runtime in success_matches:
                results.append({
                    'type': 'SUCCESS',
                    'share': float(share),
                    'exp': int(exp),
                    'runtime': float(runtime),
                    'file': filename
                })
                
            # Extract error information
            error_matches = re.findall(r'ERROR: share=([\d.]+), exp=(\d+), seed=(\d+), runtime=([\d.]+)s, error=(.+)', content)
            for share, exp, seed, runtime, error in error_matches:
                results.append({
                    'type': 'ERROR',
                    'share': float(share),
                    'exp': int(exp),
                    'seed': int(seed),
                    'runtime': float(runtime),
                    'error': error,
                    'file': filename
                })
                
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
    
    return results

def analyze_timeout_patterns():
    """Analyze timeout patterns across all log files"""
    
    # Find all SLURM output files
    log_files = glob.glob('counterfactual_slurm_logs/*.out')
    if not log_files:
        print("No log files found in counterfactual_slurm_logs/")
        return
    
    print(f"Analyzing {len(log_files)} log files...")
    
    all_results = []
    for log_file in log_files:
        results = parse_log_file(log_file)
        all_results.extend(results)
    
    # Organize by type
    timeouts = [r for r in all_results if r['type'] == 'TIMEOUT']
    successes = [r for r in all_results if r['type'] == 'SUCCESS']
    errors = [r for r in all_results if r['type'] == 'ERROR']
    
    print(f"\\n=== SUMMARY ===")
    print(f"Total experiments analyzed: {len(all_results)}")
    print(f"Successes: {len(successes)}")
    print(f"Timeouts: {len(timeouts)}")
    print(f"Errors: {len(errors)}")
    
    # Timeout analysis
    if timeouts:
        print(f"\\n=== TIMEOUT ANALYSIS ===")
        
        # Group by share value
        timeout_by_share = defaultdict(list)
        for t in timeouts:
            timeout_by_share[t['share']].append(t)
        
        for share in sorted(timeout_by_share.keys()):
            share_timeouts = timeout_by_share[share]
            print(f"\\nShare {share:.2f}: {len(share_timeouts)} timeouts")
            
            # Show timeout characteristics
            if any(t.get('last_step', 0) > 0 for t in share_timeouts):
                steps = [t['last_step'] for t in share_timeouts if t.get('last_step', 0) > 0]
                print(f"  Last steps reached: min={min(steps)}, max={max(steps)}, avg={sum(steps)/len(steps):.0f}")
            
            # Show seeds that timeout
            seeds = [t['seed'] for t in share_timeouts]
            print(f"  Timeout seeds: {sorted(seeds)[:10]}{'...' if len(seeds) > 10 else ''}")
            
            # Show performance characteristics if available
            perf_data = [t['last_perf'] for t in share_timeouts if t.get('last_perf')]
            if perf_data:
                max_sizes = [int(p[2]) for p in perf_data]
                consolidations = [int(p[4]) / (int(p[4]) + int(p[5])) for p in perf_data]
                print(f"  Max conglomerate sizes: min={min(max_sizes)}, max={max(max_sizes)}, avg={sum(max_sizes)/len(max_sizes):.0f}")
                print(f"  Consolidation rates: min={min(consolidations):.2f}, max={max(consolidations):.2f}, avg={sum(consolidations)/len(consolidations):.2f}")
    
    # Success analysis for comparison
    if successes:
        print(f"\\n=== SUCCESS ANALYSIS (for comparison) ===")
        runtimes = [s['runtime'] for s in successes]
        print(f"Success runtimes: min={min(runtimes):.1f}s, max={max(runtimes):.1f}s, avg={sum(runtimes)/len(runtimes):.1f}s")
        
        # Group by share value
        success_by_share = defaultdict(list)
        for s in successes:
            success_by_share[s['share']].append(s)
        
        for share in sorted(success_by_share.keys()):
            share_successes = success_by_share[share]
            share_runtimes = [s['runtime'] for s in share_successes]
            print(f"Share {share:.2f}: {len(share_successes)} successes, avg_runtime={sum(share_runtimes)/len(share_runtimes):.1f}s")

def analyze_simulation_completions():
    """Count how many experiments successfully completed the simulation phase"""
    
    import numpy as np
    from collections import Counter, defaultdict
    
    SHARES = np.arange(0, 0.52, 0.02)
    COUNTERFACTUALS_PER_SHARE = 50
    
    simulation_completions = Counter()
    total_experiments = Counter()
    completions_by_share = defaultdict(int)
    
    log_files = glob.glob('counterfactual_slurm_logs/*.out')
    
    print("=== SIMULATION COMPLETION ANALYSIS ===")
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
                # Extract share value from filename or content
                share_match = re.search(r'share_([\d.]+)', log_file)
                if share_match:
                    share = float(share_match.group(1))
                else:
                    continue
                
                # Count total experiments started for this share
                started_experiments = len(re.findall(r'Starting: share=', content))
                total_experiments[share] += started_experiments
                
                # Count experiments that completed simulation phase
                completed_simulations = len(re.findall(r'TIMING: Simulation completed', content))
                simulation_completions[share] += completed_simulations
                completions_by_share[share] += completed_simulations
                
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    
    print(f"Found {len(log_files)} log files")
    print()
    
    # Summary by share
    print("Simulation completion rates by share:")
    print("Share  | Started | Sim Complete | Success Rate")
    print("-------|---------|--------------|-------------")
    
    total_started = 0
    total_sim_completed = 0
    
    for share in sorted(SHARES):
        started = total_experiments[share]
        sim_completed = simulation_completions[share]
        success_rate = (sim_completed / started * 100) if started > 0 else 0
        
        total_started += started
        total_sim_completed += sim_completed
        
        print(f"{share:5.2f}  | {started:7d} | {sim_completed:12d} | {success_rate:8.1f}%")
    
    overall_sim_rate = (total_sim_completed / total_started * 100) if total_started > 0 else 0
    
    print("-------|---------|--------------|-------------")
    print(f"Total  | {total_started:7d} | {total_sim_completed:12d} | {overall_sim_rate:8.1f}%")
    print()
    
    # Identify problematic shares
    problematic_shares = []
    for share in sorted(SHARES):
        started = total_experiments[share]
        sim_completed = simulation_completions[share]
        if started > 0:
            success_rate = sim_completed / started
            if success_rate < 0.9:  # Less than 90% simulation completion
                problematic_shares.append((share, success_rate, started, sim_completed))
    
    if problematic_shares:
        print("Shares with <90% simulation completion rate:")
        for share, rate, started, completed in problematic_shares:
            print(f"  Share {share:.2f}: {completed}/{started} completed ({rate*100:.1f}%)")
    else:
        print("All shares have >90% simulation completion rate")
    
    return {
        'total_started': total_started,
        'total_sim_completed': total_sim_completed,
        'simulation_completions': dict(simulation_completions),
        'total_experiments': dict(total_experiments)
    }

def analyze_debug_progression():
    """Analyze which debug statements are reached by all vs some experiments"""
    
    # Expected configuration
    import numpy as np
    from collections import Counter
    SHARES = np.arange(0, 0.52, 0.02)
    COUNTERFACTUALS_PER_SHARE = 50
    
    # Debug statements we're looking for (in order)
    debug_statements = [
        "Starting quantiles and other calculations",
        "Computing mean_share", 
        "Computing quantiles_shares",
        "Computing max_shares",
        "Computing HHI calculations", 
        "Computing Gini coefficient with progress tracking",
        "Completed full Gini coefficient calculation",
        "Computing ranks (double sorting with progress tracking)",
        "Completed ranks calculation"
    ]
    
    # Count how many times each debug statement appears
    debug_counts = Counter()
    
    # Track both started and completed experiments
    total_started = 0
    total_completed = 0
    completions_by_share = defaultdict(int)
    
    log_files = glob.glob('counterfactual_slurm_logs/*.out')
    print(f"Analyzing {len(log_files)} log files for debug progression...")
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Count total experiments started
        started_experiments = len(re.findall(r'Starting: share=', content))
        total_started += started_experiments
        
        # Count debug statements
        for debug_stmt in debug_statements:
            count = len(re.findall(f'DEBUG: {re.escape(debug_stmt)}', content))
            debug_counts[debug_stmt] += count
        
        # Count completions by share
        completions = re.findall(r'Completed: share=([0-9.]+), exp=(\d+)', content)
        for share_str, exp_str in completions:
            share = float(share_str)
            completions_by_share[share] += 1
    
    total_completed = sum(completions_by_share.values())
    
    print(f"\n=== DEBUG STATEMENT ANALYSIS ===")
    total_expected = len(SHARES) * COUNTERFACTUALS_PER_SHARE
    
    print(f"Total expected experiments: {total_expected}")
    print(f"Total started experiments: {total_started}")
    print(f"Total completed experiments: {total_completed}")
    print(f"Missing experiments: {total_expected - total_started}")
    print(f"Failed experiments: {total_started - total_completed}")
    
    print(f"\nDebug statement progression:")
    for i, debug_stmt in enumerate(debug_statements):
        count = debug_counts[debug_stmt]
        percentage = (count / total_started * 100) if total_started > 0 else 0
        
        if count == total_started:
            status = "✅ All started experiments reach this point"
        elif count > total_started * 0.95:
            status = f"⚠️  {total_started - count} experiments fail after this"
        else:
            status = f"❌ {total_started - count} experiments fail before/during this"
        
        print(f"{i+1:2d}. {debug_stmt}")
        print(f"    Count: {count}/{total_started} ({percentage:.1f}%) {status}")
    
    # Find the last statement that all started experiments reach
    last_universal_stmt = None
    for debug_stmt in debug_statements:
        if debug_counts[debug_stmt] == total_started:
            last_universal_stmt = debug_stmt
        else:
            break
    
    if last_universal_stmt:
        print(f"\n=== FAILURE ANALYSIS ===")
        print(f"✅ All started experiments reach: '{last_universal_stmt}'")
        
        # Find first statement where failures occur
        for debug_stmt in debug_statements:
            if debug_counts[debug_stmt] < total_started:
                missing = total_started - debug_counts[debug_stmt]
                print(f"❌ First failure point: '{debug_stmt}' - {missing} experiments fail here")
                break
    else:
        print(f"\n❌ Some experiments fail before even reaching the first debug statement!")
    
    return debug_counts, completions_by_share

def analyze_incomplete_post_processing():
    """COMPLETELY NEW: Identify which chunks didn't complete certain post-processing steps"""
    
    import numpy as np
    import re
    import glob
    from collections import defaultdict
    
    # Post-processing debug patterns to track
    debug_steps = [
        "Computing market_share",
        "Computing mean_share",
        "Computing HHI calculations", 
        "Computing Gini coefficient with progress tracking",
        "Completed full Gini coefficient calculation",
        "Computing ranks (double sorting with progress tracking)",
        "Completed ranks calculation",
        "Computing conglomerate averages",
        "Assembling final model results"
    ]
    
    log_files = glob.glob('counterfactual_slurm_logs/*.out')
    print(f"=== CHUNK-LEVEL POST-PROCESSING COMPLETION ANALYSIS ===")
    print(f"Analyzing {len(log_files)} SLURM chunk log files...")
    
    if not log_files:
        print("❌ No log files found!")
        return {}, {}
    
    chunk_data = {}
    
    # Process each chunk log file
    for log_file in log_files:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Extract chunk info from filename
        share_match = re.search(r'share_([\d.]+)', log_file)
        chunk_match = re.search(r'chunk_(\d+)', log_file)
        
        if share_match:
            share = float(share_match.group(1))
            chunk_id = int(chunk_match.group(1)) if chunk_match else 0
            
            # Count experiment completions in this chunk
            started = len(re.findall(r'Starting: share=', content))
            completed = len(re.findall(r'Completed: share=', content))
            
            # Check which debug steps appear in this chunk
            debug_step_counts = {}
            for step in debug_steps:
                count = content.count(f'DEBUG: {step}')
                debug_step_counts[step] = count > 0  # Boolean: did this chunk reach this step?
            
            chunk_data[log_file] = {
                'share': share,
                'chunk_id': chunk_id,
                'started': started,
                'completed': completed,
                'success_rate': completed/started if started > 0 else 0,
                'debug_steps': debug_step_counts
            }
    
    total_started = sum(data['started'] for data in chunk_data.values())
    total_completed = sum(data['completed'] for data in chunk_data.values())
    
    print(f"TOTAL: {total_started} experiments started, {total_completed} completed ({total_completed/total_started*100:.1f}% success)")
    
    # Find chunks with problems
    problem_chunks = [(filename, data) for filename, data in chunk_data.items() 
                      if data['started'] > 0 and data['success_rate'] < 0.9]
    
    if problem_chunks:
        print(f"\\n🔍 PROBLEM CHUNKS ({len(problem_chunks)} chunks with <90% completion):")
        for filename, data in sorted(problem_chunks, key=lambda x: x[1]['share']):
            short_name = filename.split('/')[-1]
            print(f"  {short_name}: Share {data['share']:.2f}, {data['completed']}/{data['started']} completed ({data['success_rate']*100:.0f}%)")
    else:
        print("\\n✅ No problem chunks found - all chunks have >90% completion rate")
    
    # Analyze which steps are problematic
    print(f"\\n🔍 STEP-WISE COMPLETION ANALYSIS:")
    
    total_chunks = len([data for data in chunk_data.values() if data['started'] > 0])
    
    for step in debug_steps:
        chunks_reaching_step = sum(1 for data in chunk_data.values() 
                                  if data['started'] > 0 and data['debug_steps'][step])
        completion_rate = chunks_reaching_step / total_chunks if total_chunks > 0 else 0
        
        print(f"\\n{step}:")
        print(f"  Chunk completion: {chunks_reaching_step}/{total_chunks} ({completion_rate*100:.1f}%)")
        
        if completion_rate < 1.0:
            # Show which chunks failed
            failed_chunks = []
            for filename, data in chunk_data.items():
                if data['started'] > 0 and not data['debug_steps'][step]:
                    failed_chunks.append((data['share'], data['chunk_id'], filename))
            
            # Group by share
            by_share = defaultdict(list)
            for share, chunk_id, filename in failed_chunks:
                by_share[share].append(chunk_id)
            
            print("  Failed chunks by share:")
            for share in sorted(by_share.keys()):
                chunk_ids = sorted(by_share[share])
                print(f"    Share {share:.2f}: chunks {chunk_ids}")
    
    return chunk_data

def find_all_failed_experiments():
    """Find all experiments that started but did not complete successfully"""
    
    import re
    import glob
    from collections import defaultdict
    
    log_files = glob.glob('counterfactual_slurm_logs/*.out')
    
    print(f"=== FINDING ALL FAILED EXPERIMENTS ===")
    print(f"Scanning {len(log_files)} log files...")
    
    all_started = set()  # (share, exp_id)
    all_completed = set()  # (share, exp_id)
    
    chunk_details = {}  # filename -> details
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract all started experiments from this chunk
            started_matches = re.findall(r'Starting: share=([0-9.]+), exp=(\d+)', content)
            completed_matches = re.findall(r'Completed: share=([0-9.]+), exp=(\d+)', content)
            
            chunk_started = set()
            chunk_completed = set()
            
            for share_str, exp_str in started_matches:
                share = float(share_str)
                exp_id = int(exp_str)
                exp_key = (share, exp_id)
                all_started.add(exp_key)
                chunk_started.add(exp_key)
            
            for share_str, exp_str in completed_matches:
                share = float(share_str)
                exp_id = int(exp_str)
                exp_key = (share, exp_id)
                all_completed.add(exp_key)
                chunk_completed.add(exp_key)
            
            chunk_failed = chunk_started - chunk_completed
            
            chunk_details[log_file] = {
                'started': len(chunk_started),
                'completed': len(chunk_completed), 
                'failed': len(chunk_failed),
                'failed_experiments': chunk_failed
            }
            
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    
    # Calculate totals
    total_started = len(all_started)
    total_completed = len(all_completed)
    all_failed = all_started - all_completed
    
    print(f"\\nOVERALL SUMMARY:")
    print(f"  Total experiments started: {total_started}")
    print(f"  Total experiments completed: {total_completed}")
    print(f"  Total experiments failed: {len(all_failed)}")
    print(f"  Success rate: {total_completed/total_started*100:.2f}%")
    
    if all_failed:
        print(f"\\nFAILED EXPERIMENTS ({len(all_failed)} total):")
        
        # Group by share
        failed_by_share = defaultdict(list)
        for share, exp_id in sorted(all_failed):
            failed_by_share[share].append(exp_id)
        
        for share in sorted(failed_by_share.keys()):
            exp_ids = sorted(failed_by_share[share])
            print(f"  Share {share:.2f}: {len(exp_ids)} failed experiments")
            if len(exp_ids) <= 20:
                print(f"    Experiment IDs: {exp_ids}")
            else:
                print(f"    Experiment IDs: {exp_ids[:10]}...{exp_ids[-10:]}")
        
        # Show which chunks had failures
        print(f"\\nCHUNKS WITH FAILURES:")
        chunks_with_failures = [(filename, details) for filename, details in chunk_details.items() 
                               if details['failed'] > 0]
        
        for filename, details in sorted(chunks_with_failures, key=lambda x: x[1]['failed'], reverse=True):
            short_name = filename.split('/')[-1]
            print(f"  {short_name}: {details['failed']} failures ({details['completed']}/{details['started']} completed)")
            
            if details['failed'] <= 10:
                failed_list = [f"{share:.2f}:{exp_id}" for share, exp_id in sorted(details['failed_experiments'])]
                print(f"    Failed: {failed_list}")
    else:
        print("\\n✅ No failed experiments found - all started experiments completed successfully!")
    
    return all_failed, chunk_details

def analyze_specific_failures():
    """Analyze specific failure patterns and timeouts"""
    
    import numpy as np
    from collections import Counter, defaultdict
    
    log_files = glob.glob('counterfactual_slurm_logs/*.out')
    
    print("=== SPECIFIC FAILURE ANALYSIS ===")
    
    gini_failures = []
    ranks_failures = []
    timeout_failures = []
    error_failures = []
    validation_failures = []
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract share from filename
            share_match = re.search(r'share_([\d.]+)', log_file)
            if not share_match:
                continue
            share = float(share_match.group(1))
            
            # Find experiments that reach Gini but don't complete it
            gini_start = len(re.findall(r'DEBUG: Computing Gini coefficient', content))
            gini_complete = len(re.findall(r'DEBUG: Completed Gini coefficient', content))
            if gini_start > gini_complete:
                gini_failures.append((share, gini_start - gini_complete))
            
            # Find experiments that reach ranks but don't complete it
            ranks_start = len(re.findall(r'DEBUG: Computing ranks \(double sorting', content))
            ranks_complete = len(re.findall(r'DEBUG: Completed ranks calculation', content))
            if ranks_start > ranks_complete:
                ranks_failures.append((share, ranks_start - ranks_complete))
            
            # Find timeout patterns
            timeouts = re.findall(r'TIMEOUT: share=([\d.]+), exp=(\d+), seed=(\d+), runtime=([\d.]+)s', content)
            for t_share, exp, seed, runtime in timeouts:
                timeout_failures.append((float(t_share), int(exp), int(seed), float(runtime)))
            
            # Find validation errors (inf values)
            validation_errors = re.findall(r'VALIDATION ERROR: Illegal state values detected', content)
            if validation_errors:
                validation_failures.append((share, len(validation_errors)))
            
            # Find other errors
            errors = re.findall(r'ERROR: share=([\d.]+), exp=(\d+).*error=(.+)', content)
            for e_share, exp, error in errors:
                error_failures.append((float(e_share), int(exp), error.strip()))
                
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    
    # Analyze Gini failures
    if gini_failures:
        print(f"\n🔍 GINI COEFFICIENT FAILURES: {sum(f[1] for f in gini_failures)} total")
        gini_by_share = defaultdict(int)
        for share, count in gini_failures:
            gini_by_share[share] += count
        
        for share in sorted(gini_by_share.keys()):
            print(f"  Share {share:.2f}: {gini_by_share[share]} failures")
    
    # Analyze ranks failures  
    if ranks_failures:
        print(f"\n🔍 RANKS CALCULATION FAILURES: {sum(f[1] for f in ranks_failures)} total")
        ranks_by_share = defaultdict(int)
        for share, count in ranks_failures:
            ranks_by_share[share] += count
        
        for share in sorted(ranks_by_share.keys()):
            print(f"  Share {share:.2f}: {ranks_by_share[share]} failures")
    
    # Analyze timeout patterns
    if timeout_failures:
        print(f"\n⏰ TIMEOUT ANALYSIS: {len(timeout_failures)} timeouts")
        timeout_by_share = defaultdict(list)
        for share, exp, seed, runtime in timeout_failures:
            timeout_by_share[share].append(runtime)
        
        for share in sorted(timeout_by_share.keys()):
            runtimes = timeout_by_share[share]
            print(f"  Share {share:.2f}: {len(runtimes)} timeouts, avg runtime: {sum(runtimes)/len(runtimes):.1f}s")
    
    # Analyze validation failures
    if validation_failures:
        print(f"\n🚫 VALIDATION FAILURES (inf values): {sum(f[1] for f in validation_failures)} total")
        for share, count in validation_failures:
            print(f"  Share {share:.2f}: {count} validation errors")
    
    # Analyze other errors
    if error_failures:
        print(f"\n💥 OTHER ERRORS: {len(error_failures)} total")
        error_types = Counter()
        error_by_share = defaultdict(list)
        
        for share, exp, error in error_failures:
            error_types[error] += 1
            error_by_share[share].append(error)
        
        print("  Error types:")
        for error, count in error_types.most_common(5):
            print(f"    {error}: {count} times")
        
        print("  By share:")
        for share in sorted(error_by_share.keys()):
            errors = error_by_share[share]
            print(f"    Share {share:.2f}: {len(errors)} errors")
    
    return {
        'gini_failures': gini_failures,
        'ranks_failures': ranks_failures,
        'timeout_failures': timeout_failures,
        'validation_failures': validation_failures,
        'error_failures': error_failures
    }

def analyze_failure_causes():
    """Analyze WHY experiments fail - root cause analysis"""
    
    import numpy as np
    from collections import Counter, defaultdict
    
    log_files = glob.glob('counterfactual_slurm_logs/*.out')
    
    print("=== ROOT CAUSE ANALYSIS ===")
    
    # Data structures to track failure causes
    memory_issues = []
    timeout_causes = []
    computational_complexity = []
    data_characteristics = []
    system_issues = []
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract share from filename
            share_match = re.search(r'share_([\d.]+)', log_file)
            if not share_match:
                continue
            share = float(share_match.group(1))
            
            # 1. MEMORY-RELATED ISSUES
            # Check for ACTUAL memory allocation failures (very specific patterns)
            if re.search(r'MemoryError:', content):
                memory_issues.append((share, 'python_memory_error'))
            if re.search(r'out of memory|OOM|memory allocation failed', content, re.IGNORECASE):
                memory_issues.append((share, 'system_out_of_memory'))
            if re.search(r'cannot allocate.*bytes|malloc failed|allocation failure', content, re.IGNORECASE):
                memory_issues.append((share, 'allocation_failure'))
            
            # Check for large array operations before failure
            timing_data = re.findall(r'TIMING: (.+) completed in ([\d.]+)s', content)
            for operation, time_taken in timing_data:
                if float(time_taken) > 300:  # Operations taking > 5 minutes
                    computational_complexity.append((share, operation, float(time_taken)))
            
            # 2. TIMEOUT ANALYSIS - Look at what was happening when timeout occurred
            timeout_matches = re.findall(r'TIMEOUT: share=([\d.]+), exp=(\d+), seed=(\d+), runtime=([\d.]+)s', content)
            for t_share, exp, seed, runtime in timeout_matches:
                t_share_f = float(t_share)
                runtime_f = float(runtime)
                
                # Find the last debug statement before timeout
                debug_statements = [
                    "Computing mean_share", "Computing quantiles_shares", "Computing max_shares",
                    "Computing HHI calculations", "Computing Gini coefficient", 
                    "Computing cumulative sums", "Completed Gini coefficient",
                    "Computing ranks", "Completed ranks calculation"
                ]
                
                last_debug = None
                for debug_stmt in reversed(debug_statements):
                    if f'DEBUG: {debug_stmt}' in content:
                        last_debug = debug_stmt
                        break
                
                timeout_causes.append((t_share_f, runtime_f, last_debug))
            
            # 3. DATA CHARACTERISTICS - Look for patterns in successful vs failed runs
            # Extract validation data
            validation_matches = re.findall(r'VALIDATION: All state values valid, range: \[([\d.e+-]+), ([\d.e+-]+)\] \(max_finite: (True|False)\)', content)
            for min_val, max_val, is_finite in validation_matches:
                try:
                    min_v = float(min_val)
                    max_v = float(max_val) if max_val != 'inf' else float('inf')
                    finite = is_finite == 'True'
                    
                    # Track data characteristics
                    data_characteristics.append((share, min_v, max_v, finite))
                    
                    # Flag potential overflow issues
                    if not finite or max_v > 1e40:
                        memory_issues.append((share, 'numerical_overflow'))
                        
                except ValueError:
                    continue
            
            # 4. SYSTEM ISSUES
            # Check for system-level problems
            if re.search(r'killed|terminated|segmentation fault|core dumped', content, re.IGNORECASE):
                system_issues.append((share, 'system_kill'))
            
            # Check for Python/NumPy specific errors
            if re.search(r'RuntimeError|ValueError.*array|dtype', content):
                system_issues.append((share, 'numpy_error'))
                
        except Exception as e:
            print(f"Error analyzing {log_file}: {e}")
    
    # ANALYZE CAUSES
    
    print(f"\n🧠 MEMORY & OVERFLOW ISSUES:")
    if memory_issues:
        memory_by_type = defaultdict(list)
        for share, issue_type in memory_issues:
            memory_by_type[issue_type].append(share)
        
        for issue_type, shares in memory_by_type.items():
            share_summary = Counter(shares)
            print(f"  {issue_type}: {len(shares)} issues")
            for share, count in sorted(share_summary.items()):
                print(f"    Share {share:.2f}: {count} occurrences")
    else:
        print("  ✅ No explicit memory issues detected")
    
    print(f"\n⏱️  COMPUTATIONAL BOTTLENECKS:")
    if computational_complexity:
        # Group by operation type
        slow_ops = defaultdict(list)
        for share, operation, time_taken in computational_complexity:
            slow_ops[operation].append((share, time_taken))
        
        for operation, data in slow_ops.items():
            times = [t for _, t in data]
            avg_time = sum(times) / len(times)
            print(f"  {operation}: {len(data)} slow instances, avg {avg_time:.1f}s")
            
            # Show worst cases
            worst_cases = sorted(data, key=lambda x: x[1], reverse=True)[:3]
            for share, time_taken in worst_cases:
                print(f"    Share {share:.2f}: {time_taken:.1f}s")
    else:
        print("  ✅ No extremely slow operations detected")
    
    print(f"\n🎯 TIMEOUT ROOT CAUSES:")
    if timeout_causes:
        timeout_by_stage = defaultdict(list)
        timeout_by_share = defaultdict(list)
        
        for share, runtime, last_debug in timeout_causes:
            timeout_by_stage[last_debug or 'unknown'].append(runtime)
            timeout_by_share[share].append((runtime, last_debug))
        
        print("  By computation stage:")
        for stage, runtimes in timeout_by_stage.items():
            avg_runtime = sum(runtimes) / len(runtimes)
            print(f"    {stage}: {len(runtimes)} timeouts, avg runtime {avg_runtime:.1f}s")
        
        print("  By share value:")
        for share in sorted(timeout_by_share.keys()):
            timeouts = timeout_by_share[share]
            avg_runtime = sum(r for r, _ in timeouts) / len(timeouts)
            stages = [stage for _, stage in timeouts]
            stage_summary = Counter(stages)
            print(f"    Share {share:.2f}: {len(timeouts)} timeouts, avg {avg_runtime:.1f}s")
            for stage, count in stage_summary.most_common(2):
                print(f"      Mostly at: {stage} ({count} times)")
    else:
        print("  ✅ No timeout patterns detected")
    
    print(f"\n📊 DATA CHARACTERISTICS vs FAILURE PATTERNS:")
    if data_characteristics:
        # Analyze relationship between data characteristics and failures
        share_data = defaultdict(list)
        for share, min_val, max_val, is_finite in data_characteristics:
            share_data[share].append((min_val, max_val, is_finite))
        
        print("  Data ranges by share (looking for overflow patterns):")
        for share in sorted(share_data.keys()):
            data_points = share_data[share]
            max_values = [max_val for _, max_val, _ in data_points]
            finite_ratios = [is_finite for _, _, is_finite in data_points]
            
            avg_max = sum(v for v in max_values if v != float('inf')) / len([v for v in max_values if v != float('inf')]) if any(v != float('inf') for v in max_values) else float('inf')
            finite_ratio = sum(finite_ratios) / len(finite_ratios)
            
            print(f"    Share {share:.2f}: avg_max={avg_max:.2e}, finite_ratio={finite_ratio:.2f}")
            
            if finite_ratio < 0.8:
                print(f"      ⚠️  High overflow rate! {(1-finite_ratio)*100:.1f}% have inf values")
            if avg_max > 1e30:
                print(f"      ⚠️  Very large numbers detected - approaching overflow threshold")
    
    print(f"\n🔧 SYSTEM-LEVEL ISSUES:")
    if system_issues:
        issue_summary = Counter(issue_type for _, issue_type in system_issues)
        for issue_type, count in issue_summary.items():
            print(f"  {issue_type}: {count} occurrences")
            
            affected_shares = [share for share, it in system_issues if it == issue_type]
            share_summary = Counter(affected_shares)
            for share, share_count in sorted(share_summary.items()):
                print(f"    Share {share:.2f}: {share_count} times")
    else:
        print("  ✅ No system-level issues detected")
    
    return {
        'memory_issues': memory_issues,
        'computational_complexity': computational_complexity,
        'timeout_causes': timeout_causes,
        'data_characteristics': data_characteristics,
        'system_issues': system_issues
    }

def analyze_granular_hangs():
    """Analyze the new granular debug messages to find exact hang locations"""
    
    from collections import Counter, defaultdict
    
    log_files = glob.glob('counterfactual_slurm_logs/*.out')
    
    print("=== GRANULAR HANG ANALYSIS ===")
    
    gini_hangs = defaultdict(list)  # {hang_step: [(share, details), ...]}
    ranks_hangs = defaultdict(list)  # {hang_location: [(share, details), ...]}
    slow_operations = []  # [(share, operation, duration)]
    incomplete_progressions = defaultdict(list)  # {last_reached: [(share, exp), ...]}
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract share from filename
            share_match = re.search(r'share_([\d.]+)', log_file)
            if not share_match:
                continue
            share = float(share_match.group(1))
            
            # Analyze Gini coefficient hangs
            gini_steps = [
                "Step 1/5 - Sorting market share arrays",
                "Step 2/5 - Computing cumulative sums for Lorenz curve", 
                "Step 3/5 - Computing sum arrays",
                "Step 4/5 - Computing Lorenz curve",
                "Step 5/5 - Computing area under curve and Gini coefficient"
            ]
            
            # Find which Gini steps were reached vs completed
            gini_reached = []
            gini_completed = []
            
            for step in gini_steps:
                if f"DEBUG: {step}" in content:
                    gini_reached.append(step)
                
                # Check for completion messages
                if "Sorting completed in" in content:
                    gini_completed.append("Step 1/5")
                if "Cumulative sums completed in" in content:
                    gini_completed.append("Step 2/5")
                if "Sum arrays completed in" in content:
                    gini_completed.append("Step 3/5")  
                if "Lorenz curve completed in" in content:
                    gini_completed.append("Step 4/5")
                if "Final Gini calculation completed in" in content:
                    gini_completed.append("Step 5/5")
            
            # Find Gini hangs (reached but not completed)
            gini_incomplete = set(gini_reached) - set(gini_completed)
            for incomplete_step in gini_incomplete:
                gini_hangs[incomplete_step].append(share)
            
            # Analyze ranks hangs
            ranks_chunks_started = len(re.findall(r'DEBUG: Processing ranks chunk (\d+)-(\d+)', content))
            ranks_chunks_completed = len(re.findall(r'DEBUG: Completed chunk (\d+)-(\d+)', content))
            
            if ranks_chunks_started > ranks_chunks_completed:
                # Find the last chunk that was started but not completed
                started_chunks = re.findall(r'DEBUG: Processing ranks chunk (\d+)-(\d+)', content)
                completed_chunks = re.findall(r'DEBUG: Completed chunk (\d+)-(\d+)', content)
                
                for start_chunk in started_chunks:
                    if start_chunk not in completed_chunks:
                        chunk_range = f"chunk_{start_chunk[0]}-{start_chunk[1]}"
                        ranks_hangs[chunk_range].append(share)
                        break
            
            # Find slow operations (warnings)
            slow_step_matches = re.findall(r'WARNING: Step (\d+) took ([\d.]+)s - possible hang!', content)
            for step, duration in slow_step_matches:
                slow_operations.append((share, f"ranks_step_{step}", float(duration)))
            
            # Find slow Gini operations
            gini_times = re.findall(r'DEBUG: (.+) completed in ([\d.]+)s', content)
            for operation, duration in gini_times:
                duration_f = float(duration)
                if duration_f > 60:  # Operations taking >1 minute
                    slow_operations.append((share, f"gini_{operation}", duration_f))
            
            # Track overall progression to see where experiments get stuck
            if "DEBUG: Computing Gini coefficient with progress tracking" in content:
                if "DEBUG: Completed full Gini coefficient calculation" not in content:
                    incomplete_progressions["gini_incomplete"].append(share)
                    
            if "DEBUG: Computing ranks (double sorting with progress tracking)" in content:
                if "DEBUG: Completed ranks calculation" not in content:
                    incomplete_progressions["ranks_incomplete"].append(share)
                    
        except Exception as e:
            print(f"Error analyzing {log_file}: {e}")
    
    # Report findings
    print(f"\n🧮 GINI COEFFICIENT HANG ANALYSIS:")
    if gini_hangs:
        for step, shares in gini_hangs.items():
            share_counts = Counter(shares)
            print(f"  Hangs at '{step}': {len(shares)} experiments")
            for share, count in sorted(share_counts.items()):
                print(f"    Share {share:.2f}: {count} hangs")
    else:
        print("  ✅ No Gini hangs detected with new granular tracking")
    
    print(f"\n📊 RANKS CALCULATION HANG ANALYSIS:")
    if ranks_hangs:
        for location, shares in ranks_hangs.items():
            share_counts = Counter(shares)
            print(f"  Hangs at '{location}': {len(shares)} experiments")
            for share, count in sorted(share_counts.items()):
                print(f"    Share {share:.2f}: {count} hangs")
    else:
        print("  ✅ No ranks hangs detected with new granular tracking")
    
    print(f"\n⚠️  SLOW OPERATIONS DETECTED:")
    if slow_operations:
        # Group by operation type
        slow_by_operation = defaultdict(list)
        for share, operation, duration in slow_operations:
            slow_by_operation[operation].append((share, duration))
        
        for operation, data in slow_by_operation.items():
            times = [duration for _, duration in data]
            avg_time = sum(times) / len(times)
            print(f"  {operation}: {len(data)} slow instances, avg {avg_time:.1f}s")
            
            # Show worst cases
            worst_cases = sorted(data, key=lambda x: x[1], reverse=True)[:3]
            for share, duration in worst_cases:
                print(f"    Share {share:.2f}: {duration:.1f}s")
    else:
        print("  ✅ No abnormally slow operations detected")
    
    print(f"\n📈 INCOMPLETE PROGRESSION SUMMARY:")
    if incomplete_progressions:
        for stage, shares in incomplete_progressions.items():
            share_counts = Counter(shares)
            print(f"  {stage}: {len(shares)} experiments incomplete")
            for share, count in sorted(share_counts.items()):
                print(f"    Share {share:.2f}: {count} experiments")
    else:
        print("  ✅ All experiments with granular tracking completed successfully")
    
    return {
        'gini_hangs': dict(gini_hangs),
        'ranks_hangs': dict(ranks_hangs),
        'slow_operations': slow_operations,
        'incomplete_progressions': dict(incomplete_progressions)
    }

def analyze_illegal_values():
    """Check for illegal values (NaN, Inf, negative) that could cause numerical instability"""
    
    from collections import Counter, defaultdict
    
    log_files = glob.glob('counterfactual_slurm_logs/*.out')
    if not log_files:
        print("No log files found!")
        return {}
    
    print(f"\n=== ILLEGAL VALUES ANALYSIS ===")
    print(f"Analyzing {len(log_files)} log files for illegal values...")
    
    # Track validation results
    validation_results = defaultdict(list)
    overflow_patterns = defaultdict(list)
    error_patterns = defaultdict(list)
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Extract share value from filename
            share_match = re.search(r'share_([\d.]+)_', log_file)
            share = float(share_match.group(1)) if share_match else None
            
            # Check for validation errors (from collaborative_growth.py validation)
            validation_errors = re.findall(r'VALIDATION ERROR: Illegal state values detected - inf: (True|False), nan: (True|False), negative: (True|False), range: \[([\d.e+-]+), ([\d.e+-]+)\]', content)
            for has_inf, has_nan, has_negative, min_val, max_val in validation_errors:
                validation_results[share].append({
                    'has_inf': has_inf == 'True',
                    'has_nan': has_nan == 'True', 
                    'has_negative': has_negative == 'True',
                    'min_val': float(min_val),
                    'max_val': float(max_val),
                    'file': log_file
                })
            
            # Check for overflow analysis (from our new overflow detection)
            overflow_matches = re.findall(r'OVERFLOW ANALYSIS: (\d+) infinite values, (\d+) finite values', content)
            for n_inf, n_finite in overflow_matches:
                overflow_patterns[share].append({
                    'n_infinite': int(n_inf),
                    'n_finite': int(n_finite),
                    'file': log_file
                })
            
            # Check for largest finite values before overflow
            largest_finite = re.findall(r'OVERFLOW ANALYSIS: Largest finite state: ([\d.e+-]+)', content)
            smallest_finite = re.findall(r'OVERFLOW ANALYSIS: Smallest positive state: ([\d.e+-]+)', content)
            
            if largest_finite and smallest_finite:
                overflow_patterns[share].append({
                    'largest_finite': float(largest_finite[-1]),
                    'smallest_finite': float(smallest_finite[-1]),
                    'file': log_file
                })
            
            # Check for specific numerical errors
            numerical_errors = re.findall(r'(RuntimeWarning.*invalid value|RuntimeWarning.*overflow|RuntimeWarning.*divide by zero)', content)
            for error in numerical_errors:
                error_patterns[share].append({
                    'error': error.strip(),
                    'file': log_file
                })
            
            # Check for problematic array warnings
            array_warnings = re.findall(r'WARNING: (.*NaN.*|.*Inf.*|.*problematic values.*)', content)
            for warning in array_warnings:
                error_patterns[share].append({
                    'warning': warning.strip(),
                    'file': log_file
                })
                
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    
    # Condensed Summary Report
    total_validation_errors = sum(len(errors) for errors in validation_results.values())
    total_overflow_issues = sum(len(patterns) for patterns in overflow_patterns.values()) 
    total_numerical_errors = sum(len(errors) for errors in error_patterns.values())
    
    print(f"\n=== ILLEGAL VALUES SUMMARY ===")
    print(f"📊 Total: {total_validation_errors} validation errors, {total_overflow_issues} overflow issues, {total_numerical_errors} numerical warnings")
    
    if total_validation_errors > 0:
        print(f"\n🚨 VALIDATION ERRORS ({total_validation_errors} total):")
        problem_shares = []
        for share in sorted(validation_results.keys()):
            errors = validation_results[share]
            error_types = set()
            for error in errors:
                if error['has_inf']: error_types.add("INF")
                if error['has_nan']: error_types.add("NaN") 
                if error['has_negative']: error_types.add("NEG")
            problem_shares.append(f"{share:.2f}({'/'.join(error_types)})")
        print(f"   Problem shares: {', '.join(problem_shares)}")
    
    if total_overflow_issues > 0:
        print(f"\n⚠️  OVERFLOW ISSUES ({total_overflow_issues} total):")
        overflow_shares = []
        largest_values = []
        for share in sorted(overflow_patterns.keys()):
            patterns = overflow_patterns[share]
            max_ratio = 0
            max_value = 0
            for pattern in patterns:
                if 'largest_finite' in pattern:
                    ratio = pattern['largest_finite'] / pattern['smallest_finite'] if pattern['smallest_finite'] > 0 else float('inf')
                    max_ratio = max(max_ratio, ratio)
                    max_value = max(max_value, pattern['largest_finite'])
            if max_ratio > 0:
                overflow_shares.append(f"{share:.2f}(ratio:{max_ratio:.1e})")
                largest_values.append(f"{share:.2f}(max:{max_value:.1e})")
        
        if overflow_shares:
            print(f"   High ratios: {', '.join(overflow_shares)}")
        if largest_values:
            print(f"   Largest values: {', '.join(largest_values)}")
    
    if total_numerical_errors > 0:
        print(f"\n⚠️  NUMERICAL WARNINGS ({total_numerical_errors} total):")
        error_counts = Counter()
        for share in error_patterns:
            for error in error_patterns[share]:
                if 'error' in error:
                    error_counts[error['error']] += 1
                elif 'warning' in error:
                    error_counts[error['warning']] += 1
        
        top_errors = error_counts.most_common(3)
        for error, count in top_errors:
            short_error = error.split(':')[-1].strip() if ':' in error else error
            print(f"   {count}x: {short_error}")
    
    if total_validation_errors == 0 and total_overflow_issues == 0 and total_numerical_errors == 0:
        print("✅ No illegal values detected - all experiments completed with legal state values!")
    
    return {
        'validation_errors': dict(validation_results),
        'overflow_patterns': dict(overflow_patterns),
        'error_patterns': dict(error_patterns)
    }

def analyze_performance_root_causes():
    """Analyze WHY certain post-processing instances are slow/fail - root cause performance analysis"""
    
    from collections import Counter, defaultdict
    import numpy as np
    
    log_files = glob.glob('counterfactual_slurm_logs/*.out')
    
    print("=== POST-PROCESSING PERFORMANCE ROOT CAUSE ANALYSIS ===")
    
    # Data structures for analysis
    timing_data = defaultdict(list)  # {share: [(runtime, success/fail), ...]}
    performance_patterns = defaultdict(list)  # {share: [simulation_characteristics, ...]}
    failure_characteristics = []  # [(share, exp, failure_mode, timing)]
    success_characteristics = []  # [(share, exp, runtime, timing_breakdown)]
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract share from filename
            share_match = re.search(r'share_([\d.]+)', log_file)
            if not share_match:
                continue
            share = float(share_match.group(1))
            
            # Get simulation timing (this always succeeds)
            sim_times = re.findall(r'TIMING: Simulation completed in ([\d.]+)s', content)
            sim_time = float(sim_times[0]) if sim_times else None
            
            # Get post-processing timing breakdown
            postproc_times = re.findall(r'TIMING: Post-processing completed in ([\d.]+)s', content)
            postproc_time = float(postproc_times[0]) if postproc_times else None
            
            # Get total runtimes for success vs timeout
            successes = re.findall(r'Completed: share=([\d.]+), exp=(\d+), runtime=([\d.]+)s', content)
            timeouts = re.findall(r'TIMEOUT: share=([\d.]+), exp=(\d+), seed=(\d+), runtime=([\d.]+)s', content)
            
            # Analyze successful experiments
            for s_share, exp, runtime in successes:
                if float(s_share) == share:
                    runtime_f = float(runtime)
                    timing_data[share].append((runtime_f, 'success'))
                    
                    # Get detailed timing breakdown for this experiment
                    timing_breakdown = {
                        'simulation_time': sim_time,
                        'postprocessing_time': postproc_time,
                        'total_runtime': runtime_f
                    }
                    success_characteristics.append((share, int(exp), runtime_f, timing_breakdown))
            
            # Analyze timeout experiments  
            for t_share, exp, seed, runtime in timeouts:
                if float(t_share) == share:
                    runtime_f = float(runtime)
                    timing_data[share].append((runtime_f, 'timeout'))
                    
                    # Determine where timeout occurred
                    failure_mode = 'unknown'
                    if 'DEBUG: Computing Gini coefficient' in content:
                        if 'DEBUG: Completed full Gini coefficient' not in content:
                            failure_mode = 'gini_timeout'
                        elif 'DEBUG: Computing ranks' in content:
                            if 'DEBUG: Completed ranks calculation' not in content:
                                failure_mode = 'ranks_timeout'
                            else:
                                failure_mode = 'late_postproc_timeout'
                    
                    failure_characteristics.append((share, int(exp), failure_mode, runtime_f))
            
            # Extract simulation characteristics that might affect post-processing
            # Look for validation data (state value ranges)
            validation_matches = re.findall(r'VALIDATION: All state values valid, range: \[([\d.e+-]+), ([\d.e+-]+)\].*max_finite: (True|False)', content)
            for min_val, max_val, is_finite in validation_matches:
                try:
                    min_v = float(min_val)
                    max_v = float(max_val) if max_val != 'inf' else float('inf')
                    finite = is_finite == 'True'
                    
                    characteristics = {
                        'min_state_value': min_v,
                        'max_state_value': max_v,
                        'state_finite': finite,
                        'state_range': max_v - min_v if finite else float('inf'),
                        'simulation_time': sim_time
                    }
                    performance_patterns[share].append(characteristics)
                    
                except ValueError:
                    continue
                    
        except Exception as e:
            print(f"Error analyzing {log_file}: {e}")
    
    # ANALYSIS AND REPORTING
    print(f"\n🎯 PERFORMANCE TIMING ANALYSIS:")
    
    # Calculate performance statistics by share
    perf_summary = {}
    for share in sorted(timing_data.keys()):
        runtimes = [runtime for runtime, status in timing_data[share]]
        successes = [runtime for runtime, status in timing_data[share] if status == 'success']
        timeouts = [runtime for runtime, status in timing_data[share] if status == 'timeout']
        
        if runtimes:
            perf_summary[share] = {
                'total_experiments': len(runtimes),
                'successes': len(successes),
                'timeouts': len(timeouts),
                'success_rate': len(successes) / len(runtimes),
                'avg_runtime_success': np.mean(successes) if successes else None,
                'avg_runtime_timeout': np.mean(timeouts) if timeouts else None,
                'max_runtime_success': np.max(successes) if successes else None
            }
    
    # Print performance summary
    print("Share  | Succ | Timeout | Succ Rate | Avg Success | Avg Timeout | Max Success")
    print("-------|------|---------|-----------|-------------|-------------|------------")
    
    for share in sorted(perf_summary.keys()):
        p = perf_summary[share]
        avg_succ = f"{p['avg_runtime_success']:8.0f}s" if p['avg_runtime_success'] else "      N/A"
        avg_timeout = f"{p['avg_runtime_timeout']:8.0f}s" if p['avg_runtime_timeout'] else "      N/A"
        max_succ = f"{p['max_runtime_success']:8.0f}s" if p['max_runtime_success'] else "      N/A"
        
        print(f"{share:5.2f}  | {p['successes']:4d} | {p['timeouts']:7d} | {p['success_rate']:8.1%} | {avg_succ} | {avg_timeout} | {max_succ}")
    
    print(f"\n💡 KEY INSIGHTS:")
    
    # Insight 1: Performance degradation with share value
    success_rates = [(share, perf_summary[share]['success_rate']) for share in sorted(perf_summary.keys())]
    if success_rates:
        print("  Performance vs Share Value:")
        low_success = [share for share, rate in success_rates if rate < 0.8]
        if low_success:
            print(f"    Shares with <80% success rate: {low_success}")
            print(f"    Pattern: Higher shares = more timeouts (computational complexity increases)")
    
    # Insight 2: Runtime distribution analysis
    all_success_runtimes = [runtime for share_data in timing_data.values() 
                           for runtime, status in share_data if status == 'success']
    all_timeout_runtimes = [runtime for share_data in timing_data.values() 
                           for runtime, status in share_data if status == 'timeout']
    
    if all_success_runtimes and all_timeout_runtimes:
        print(f"  Runtime Distribution:")
        print(f"    Successful experiments: {np.mean(all_success_runtimes):.0f}s avg (range: {np.min(all_success_runtimes):.0f}-{np.max(all_success_runtimes):.0f}s)")
        print(f"    Timeout experiments: {np.mean(all_timeout_runtimes):.0f}s avg (all hit timeout limit)")
        print(f"    Gap: ~{np.mean(all_timeout_runtimes) - np.max(all_success_runtimes):.0f}s between max success and timeout")
    
    # Insight 3: Data characteristics correlation
    if performance_patterns:
        print(f"  Data Characteristics vs Performance:")
        
        # Analyze state value ranges for different shares
        high_performing_shares = [share for share, p in perf_summary.items() if p['success_rate'] > 0.9]
        low_performing_shares = [share for share, p in perf_summary.items() if p['success_rate'] < 0.7]
        
        if high_performing_shares and low_performing_shares:
            print(f"    High-performing shares (>90% success): {high_performing_shares[:3]}")
            print(f"    Low-performing shares (<70% success): {low_performing_shares}")
            
            # Compare characteristics
            high_chars = [char for share in high_performing_shares[:3] 
                         for char in performance_patterns.get(share, [])]
            low_chars = [char for share in low_performing_shares 
                        for char in performance_patterns.get(share, [])]
            
            if high_chars and low_chars:
                high_max_states = [c['max_state_value'] for c in high_chars if np.isfinite(c['max_state_value'])]
                low_max_states = [c['max_state_value'] for c in low_chars if np.isfinite(c['max_state_value'])]
                
                if high_max_states and low_max_states:
                    print(f"    High-perf max state values: avg {np.mean(high_max_states):.2e}")
                    print(f"    Low-perf max state values: avg {np.mean(low_max_states):.2e}")
                    print(f"    → State value magnitude correlates with post-processing difficulty")
    
    # Insight 4: Failure mode analysis
    if failure_characteristics:
        failure_modes = Counter([mode for _, _, mode, _ in failure_characteristics])
        print(f"  Failure Mode Breakdown:")
        for mode, count in failure_modes.most_common():
            print(f"    {mode}: {count} timeouts")
    
    print(f"\n🔧 RECOMMENDED ACTIONS:")
    print("  1. For high-share timeouts: Implement adaptive chunking based on state value magnitude")
    print("  2. For ranks timeouts: Consider approximate ranking algorithms for very large values")  
    print("  3. For Gini timeouts: Implement progressive downsampling for extreme distributions")
    print("  4. Set timeout limits based on share value (higher shares = longer timeouts)")
    
    return {
        'timing_data': dict(timing_data),
        'performance_summary': perf_summary,
        'failure_characteristics': failure_characteristics,
        'success_characteristics': success_characteristics,
        'performance_patterns': dict(performance_patterns)
    }

def debug_hang_detection():
    """Debug what's being falsely detected as hangs"""
    log_files = glob.glob('counterfactual_slurm_logs/*.out')[:3]  # Just check first 3 files
    
    for log_file in log_files:
        print(f"\nChecking {log_file}:")
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Check if this file has completed experiments
        completions = len(re.findall(r'Completed: share=', content))
        print(f"  Completed experiments: {completions}")
        
        # Check Gini tracking
        gini_start = content.count("DEBUG: Computing Gini coefficient with progress tracking")
        gini_complete = content.count("DEBUG: Completed full Gini coefficient calculation")
        print(f"  Gini: {gini_start} started, {gini_complete} completed")
        
        # Check ranks tracking  
        ranks_start = content.count("DEBUG: Computing ranks (double sorting with progress tracking)")
        ranks_complete = content.count("DEBUG: Completed ranks calculation")
        print(f"  Ranks: {ranks_start} started, {ranks_complete} completed")
        
        # Show sample of relevant lines
        gini_lines = [line.strip() for line in content.split('\n') if 'Gini coefficient' in line]
        if gini_lines:
            print("  Sample Gini lines:")
            for line in gini_lines[:3]:
                print(f"    {line}")

def debug_memory_detection():
    """Debug what's being detected as memory issues"""
    log_files = glob.glob('counterfactual_slurm_logs/*.out')[:2]  # Just check first 2 files
    
    for log_file in log_files:
        print(f"\nChecking {log_file}:")
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Look for all lines containing "memory"
        memory_lines = [line.strip() for line in content.split('\n') if 'memory' in line.lower()]
        print(f"Found {len(memory_lines)} lines with 'memory':")
        for line in memory_lines[:5]:  # Show first 5
            print(f"  {line}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        analyze_debug_progression()
    elif len(sys.argv) > 1 and sys.argv[1] == "simulation":
        analyze_simulation_completions()
    elif len(sys.argv) > 1 and sys.argv[1] == "failures":
        analyze_specific_failures()
    elif len(sys.argv) > 1 and sys.argv[1] == "causes":
        analyze_failure_causes()
    elif len(sys.argv) > 1 and sys.argv[1] == "memory_debug":
        debug_memory_detection()
    elif len(sys.argv) > 1 and sys.argv[1] == "granular":
        analyze_granular_hangs()
    elif len(sys.argv) > 1 and sys.argv[1] == "hang_debug":
        debug_hang_detection()
    elif len(sys.argv) > 1 and sys.argv[1] == "performance":
        analyze_performance_root_causes()
    elif len(sys.argv) > 1 and sys.argv[1] == "illegal":
        analyze_illegal_values()
    elif len(sys.argv) > 1 and sys.argv[1] == "incomplete":
        analyze_incomplete_post_processing()
    elif len(sys.argv) > 1 and sys.argv[1] == "failed":
        find_all_failed_experiments()
    else:
        # Run all analyses by default
        print("=== TIMEOUT ANALYSIS ===")
        analyze_timeout_patterns()
        print("\n")
        analyze_simulation_completions()
        print("\n=== DEBUG PROGRESSION ANALYSIS ===")
        analyze_debug_progression()
        print("\n")
        analyze_performance_root_causes()