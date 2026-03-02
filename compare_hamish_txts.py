def parse_file1(filename):
    """Parse MacawAllDirsByUniqueBird.txt"""
    birds = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',', 3)  # Split into max 4 parts
            if len(parts) >= 4:
                bird_id = parts[0]
                code = parts[1]
                file_types = parts[2]
                paths_str = parts[3]

                # Count paths by splitting on '//' and filtering empty strings
                paths = [p.strip() for p in paths_str.split('//') if p.strip()]

                birds[bird_id] = {
                    'code': code,
                    'file_types': file_types,
                    'path_count': len(paths),
                    'paths': set(paths)  # Use set for easier comparison
                }
    return birds


def parse_file2(filename):
    """Parse MacawAllDirByBird.txt"""
    birds = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',', 3)  # Split into max 4 parts
            if len(parts) >= 4:
                bird_id = parts[0]
                code = parts[1]
                file_type = parts[2]
                paths_str = parts[3]

                # Extract paths from brackets and split by '//'
                if paths_str.startswith('[') and paths_str.endswith(']'):
                    paths_content = paths_str[1:-1]  # Remove brackets
                    paths = [p.strip() for p in paths_content.split('//') if p.strip()]
                else:
                    paths = [paths_str.strip()] if paths_str.strip() else []

                birds[bird_id] = {
                    'code': code,
                    'file_type': file_type,
                    'path_count': len(paths),
                    'paths': set(paths)  # Use set for easier comparison
                }
    return birds


def normalize_path(path):
    """Normalize path for comparison (handle different separators, etc.)"""
    # Convert backslashes to forward slashes and remove extra spaces
    return path.replace('\\', '/').strip()


def compare_paths(paths1, paths2):
    """Compare two sets of paths and return overlap statistics"""
    # Normalize paths for comparison
    norm_paths1 = {normalize_path(p) for p in paths1}
    norm_paths2 = {normalize_path(p) for p in paths2}

    overlap = norm_paths1 & norm_paths2
    only_in_1 = norm_paths1 - norm_paths2
    only_in_2 = norm_paths2 - norm_paths1

    return {
        'overlap': overlap,
        'only_in_1': only_in_1,
        'only_in_2': only_in_2,
        'overlap_count': len(overlap),
        'total_unique': len(norm_paths1 | norm_paths2)
    }


def compare_files(file1_path, file2_path):
    """Compare the two files and generate a detailed report"""

    print("Parsing files...")
    birds1 = parse_file1(file1_path)
    birds2 = parse_file2(file2_path)

    print(f"\n=== FILE COMPARISON REPORT ===")
    print(f"File 1 (MacawAllDirsByUniqueBird.txt): {len(birds1)} birds")
    print(f"File 2 (MacawAllDirByBird.txt): {len(birds2)} birds")

    # Find birds in each file
    birds1_set = set(birds1.keys())
    birds2_set = set(birds2.keys())

    only_in_file1 = birds1_set - birds2_set
    only_in_file2 = birds2_set - birds1_set
    in_both = birds1_set & birds2_set

    print(f"\n=== BIRD DISTRIBUTION ===")
    print(f"Birds only in File 1: {len(only_in_file1)}")
    print(f"Birds only in File 2: {len(only_in_file2)}")
    print(f"Birds in both files: {len(in_both)}")

    # Show birds only in each file
    if only_in_file1:
        print(f"\n=== BIRDS ONLY IN FILE 1 ===")
        for bird in sorted(only_in_file1):
            print(f"  {bird} ({birds1[bird]['path_count']} paths)")

    if only_in_file2:
        print(f"\n=== BIRDS ONLY IN FILE 2 ===")
        for bird in sorted(only_in_file2):
            print(f"  {bird} ({birds2[bird]['path_count']} paths)")

    # Compare birds that appear in both files - PATH CONTENT ANALYSIS
    if in_both:
        print(f"\n=== BIRDS IN BOTH FILES - PATH CONTENT COMPARISON ===")

        identical_paths = []
        different_paths = []

        for bird in sorted(in_both):
            paths1 = birds1[bird]['paths']
            paths2 = birds2[bird]['paths']

            comparison = compare_paths(paths1, paths2)

            if comparison['overlap_count'] == len(paths1) == len(paths2) and comparison['total_unique'] == len(paths1):
                # Identical paths
                identical_paths.append(bird)
            else:
                # Different paths
                different_paths.append((bird, comparison))

        print(f"\nBirds with IDENTICAL paths: {len(identical_paths)}")
        if len(identical_paths) <= 10:  # Show all if 10 or fewer
            for bird in identical_paths:
                print(f"  {bird}")
        else:
            print(f"  (Too many to list - {len(identical_paths)} birds)")

        print(f"\nBirds with DIFFERENT paths: {len(different_paths)}")

        if different_paths:
            print("\nDetailed path differences:")
            print("=" * 80)

            for bird, comp in different_paths:
                count1 = birds1[bird]['path_count']
                count2 = birds2[bird]['path_count']

                print(f"\n{bird}:")
                print(
                    f"  File 1: {count1} paths | File 2: {count2} paths | Overlap: {comp['overlap_count']} | Total unique: {comp['total_unique']}")

                if comp['only_in_1']:
                    print(f"  Paths ONLY in File 1 ({len(comp['only_in_1'])}):")
                    for path in sorted(list(comp['only_in_1'])[:3]):  # Show first 3
                        print(f"    {path}")
                    if len(comp['only_in_1']) > 3:
                        print(f"    ... and {len(comp['only_in_1']) - 3} more")

                if comp['only_in_2']:
                    print(f"  Paths ONLY in File 2 ({len(comp['only_in_2'])}):")
                    for path in sorted(list(comp['only_in_2'])[:3]):  # Show first 3
                        print(f"    {path}")
                    if len(comp['only_in_2']) > 3:
                        print(f"    ... and {len(comp['only_in_2']) - 3} more")

                        if comp['overlap']:
                            print(f"  Shared paths ({len(comp['overlap'])}):")
                            for path in sorted(list(comp['overlap'])[:2]):  # Show first 2
                                print(f"    {path}")
                            if len(comp['overlap']) > 2:
                                print(f"    ... and {len(comp['overlap']) - 2} more")

    # Summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    if birds1:
        avg_paths1 = sum(bird['path_count'] for bird in birds1.values()) / len(birds1)
        max_paths1 = max(bird['path_count'] for bird in birds1.values())
        print(f"File 1 - Average paths per bird: {avg_paths1:.1f}, Max paths: {max_paths1}")

    if birds2:
        avg_paths2 = sum(bird['path_count'] for bird in birds2.values()) / len(birds2)
        max_paths2 = max(bird['path_count'] for bird in birds2.values())
        print(f"File 2 - Average paths per bird: {avg_paths2:.1f}, Max paths: {max_paths2}")

    # Overall path overlap summary
    if in_both:
        total_identical = len(identical_paths)
        total_different = len(different_paths)
        identical_percentage = (total_identical / len(in_both)) * 100

        print(f"\n=== PATH OVERLAP SUMMARY ===")
        print(
            f"Birds with identical paths: {total_identical}/{len(in_both)} ({identical_percentage:.1f}%)")
        print(
            f"Birds with different paths: {total_different}/{len(in_both)} ({100 - identical_percentage:.1f}%)")


def detailed_bird_analysis(bird_id, file1_path, file2_path):
    """Analyze a specific bird in detail"""
    birds1 = parse_file1(file1_path)
    birds2 = parse_file2(file2_path)

    if bird_id not in birds1 and bird_id not in birds2:
        print(f"Bird {bird_id} not found in either file.")
        return

    print(f"\n=== DETAILED ANALYSIS FOR {bird_id} ===")

    if bird_id in birds1:
        print(f"File 1: {birds1[bird_id]['path_count']} paths")
        for i, path in enumerate(sorted(birds1[bird_id]['paths']), 1):
            print(f"  {i:2d}. {path}")
    else:
        print("File 1: Bird not found")

    print()

    if bird_id in birds2:
        print(f"File 2: {birds2[bird_id]['path_count']} paths")
        for i, path in enumerate(sorted(birds2[bird_id]['paths']), 1):
            print(f"  {i:2d}. {path}")
    else:
        print("File 2: Bird not found")

    if bird_id in birds1 and bird_id in birds2:
        comparison = compare_paths(birds1[bird_id]['paths'], birds2[bird_id]['paths'])
        print(f"\nPath comparison:")
        print(f"  Overlapping paths: {comparison['overlap_count']}")
        print(f"  Only in File 1: {len(comparison['only_in_1'])}")
        print(f"  Only in File 2: {len(comparison['only_in_2'])}")
        print(f"  Total unique paths: {comparison['total_unique']}")


# Run the comparison
if __name__ == "__main__":
    # Replace these with your actual file paths
    file1 = "MacawAllDirsByUniqueBird.txt"
    file2 = "MacawAllDirByBird.txt"

    try:
        compare_files(file1, file2)

        # Uncomment the line below to analyze a specific bird in detail
        # detailed_bird_analysis("or13or14", file1, file2)

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"Error: {e}")

# Run the comparison
if __name__ == "__main__":
    # Replace these with your actual file paths
    import os
    cd = os.getcwd()
    file1 = os.path.join(cd, 'refs', 'MacawAllDirsByUniqueBird.txt')
    file2 = os.path.join(cd, 'refs', 'MacawAllDirsByBird.txt')

    try:
        compare_files(file1, file2)
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"Error: {e}")