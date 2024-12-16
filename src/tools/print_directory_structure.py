import os
import sys

def get_input_directory():
    while True:
        if len(sys.argv) > 1:
            input_directory = sys.argv[1]
        else:
            user_input = input('Specify the directory for which you wish to print its structure or press Enter to use the current directory: ')
            if user_input == '':
                return os.getcwd()
            else:
                input_directory = user_input
        
        # Validate the provided path
        if not os.path.isdir(input_directory):
            print("The specified path is not a directory. Please try again.")
            continue
        else:
            return input_directory

def print_directory_structure(root_dir):
    for root, dirs, files in os.walk(root_dir):
        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

if __name__ == "__main__":
    input_directory = get_input_directory()
    print_directory_structure(input_directory)