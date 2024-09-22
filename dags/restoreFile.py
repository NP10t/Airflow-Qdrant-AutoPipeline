from utils import *

# tại vì sau khi load lên database thay vì xóa cả thư mục và file đi 
# thì mình sẽ đổi tên file thành tên file_deleted.csv
# và để phục hồi file thì mình sẽ đổi tên file từ tên file_deleted.csv thành tên file.csv

# Sử dụng hàm để phục hồi các file
restore_deleted_files_in_directory(getFramesPath())