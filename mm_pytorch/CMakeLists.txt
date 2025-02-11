add_library(python_interface SHARED python_interface.cu)

# Ensure position-independent code (PIC) is enabled for shared libraries
set_target_properties(python_interface PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Add the required compiler flags
target_compile_options(python_interface PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>)