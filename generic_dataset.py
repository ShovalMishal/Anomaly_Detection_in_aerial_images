import os.path

from torchvision.datasets import ImageFolder


def is_valid_file_wrapper(path_to_exclusion_file):
    with open(path_to_exclusion_file, 'r') as f:
        data = f.read()

    def is_valid_file(filename):
        return os.path.basename(filename) not in data.splitlines()

    return is_valid_file


def main():
    full_dataset = ImageFolder(root='/home/shoval/Desktop/main_dataset/')
    exclusion_file = '/home/shoval/Desktop/exclusion_list.txt'
    trimmed_dataset = ImageFolder(root='/home/shoval/Desktop/main_dataset/',
                                  is_valid_file=is_valid_file_wrapper(exclusion_file))
    for idx, sample in enumerate(full_dataset):
        print(idx, sample)
    print("~" * 10)
    for idx, sample in enumerate(trimmed_dataset):
        print(idx, sample)




if __name__ == '__main__':
    main()
