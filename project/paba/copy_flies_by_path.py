
import os
import shutil
def copy_and_rename_files(src_folder, dest_folder, file_names, new_names):

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    src_file = os.path.join(src_folder, file_names)
    dest_file = os.path.join(dest_folder, new_names)
    shutil.copy(src_file, dest_file)

if __name__ == "__main__":
    # Example usage
    src_folder = "path/to/source/folder"
    dest_folder = "path/to/destination/folder"
    file_names = ["file1.txt", "file3.pdf", "file5.jpg"]  # List of file names to be copied
    new_names = ["new_file1.txt", "new_file3.pdf", "new_file5.jpg"]  # Corresponding new names
    copy_and_rename_files(src_folder, dest_folder, file_names, new_names)



















# import os
# import shutil
#
# def copy_files_by_name(src_folder, dest_folder, file_names):
#     """
#     Copies files from the source folder to the destination folder based on a list of file names.
#
#     :param src_folder: Path to the source folder
#     :param dest_folder: Path to the destination folder
#     :param file_names: List of file names to be copied
#     """
#     # Get a list of all files in the source folder
#     files = os.listdir(src_folder)
#
#     # Ensure the destination folder exists
#     if not os.path.exists(dest_folder):
#         os.makedirs(dest_folder)
#
#     # Iterate over the file names list and copy each file
#     for file_name in file_names:
#         # If the file exists in the source folder, copy it
#         if file_name in files:
#             src_file = os.path.join(src_folder, file_name)
#             dest_file = os.path.join(dest_folder, file_name)
#             try:
#                 shutil.copy(src_file, dest_file)
#                 print(f"File {file_name} has been successfully copied to {dest_folder}")
#             except Exception as e:
#                 print(f"Error copying file {file_name}: {e}")
#         else:
#             print(f"File {file_name} not found in the source folder.")
#
# # Example usage
# src_folder = "path/to/source/folder"
# dest_folder = "path/to/destination/folder"
# file_names = ["file1.txt", "file3.pdf", "file5.jpg"]  # List of file names to be copied
#
# copy_files_by_name(src_folder, dest_folder, file_names)