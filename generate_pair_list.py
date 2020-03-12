from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path

parser = ArgumentParser(description='image pair generator from two subsamplings of the same video',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_folder1', metavar='PATH',
                    help='path to folder or subsampled video', type=Path)
parser.add_argument('--input_folder2', metavar='PATH',
                    help='path to folder of full-sampled video', type=Path)
parser.add_argument('--range', metavar='R', default=10, type=int,
                    help='')
parser.add_argument('--output', metavar='DIR', default="pairs.txt", type=Path)


def generate_pairs(list1, list2, matching_range):
    pairs = []
    for n, img in enumerate(list1):
        percentage = n/len(list1)
        corresponding_index = int(percentage * len(list2))
        start = max(0, corresponding_index - matching_range//2)
        end = min(len(list2) - 1, corresponding_index + matching_range//2)
        for j in range(start, end):
            pairs.append("{} {}\n".format(img, list2[j]))
    return pairs


def main():
    args = parser.parse_args()
    with open(args.input_folder1/"images.txt", 'r') as f:
        list1 = f.read().splitlines()
    with open(args.input_folder2/"images.txt", 'r') as f:
        list2 = f.read().splitlines()
    pairs = generate_pairs(list1, list2, args.range)

    with open(args.output, 'w') as f:
        f.writelines(pairs)


if __name__ == '__main__':
    main()
