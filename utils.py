def check_path_name(file_path):
    if file_path[-1] != "/":
        return file_path+"/"
    else:
        return file_path