# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/i82carij/3º/FSIV_en/Practical_assignment/backg_moodle_v01

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/i82carij/3º/FSIV_en/Practical_assignment/backg_moodle_v01/build

# Include any dependencies generated for this target.
include CMakeFiles/vseg.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/vseg.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/vseg.dir/flags.make

CMakeFiles/vseg.dir/vseg.cc.o: CMakeFiles/vseg.dir/flags.make
CMakeFiles/vseg.dir/vseg.cc.o: ../vseg.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/i82carij/3º/FSIV_en/Practical_assignment/backg_moodle_v01/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/vseg.dir/vseg.cc.o"
	/usr/local/opt/gcc-5.3.0-32/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vseg.dir/vseg.cc.o -c /home/i82carij/3º/FSIV_en/Practical_assignment/backg_moodle_v01/vseg.cc

CMakeFiles/vseg.dir/vseg.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vseg.dir/vseg.cc.i"
	/usr/local/opt/gcc-5.3.0-32/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/i82carij/3º/FSIV_en/Practical_assignment/backg_moodle_v01/vseg.cc > CMakeFiles/vseg.dir/vseg.cc.i

CMakeFiles/vseg.dir/vseg.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vseg.dir/vseg.cc.s"
	/usr/local/opt/gcc-5.3.0-32/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/i82carij/3º/FSIV_en/Practical_assignment/backg_moodle_v01/vseg.cc -o CMakeFiles/vseg.dir/vseg.cc.s

# Object files for target vseg
vseg_OBJECTS = \
"CMakeFiles/vseg.dir/vseg.cc.o"

# External object files for target vseg
vseg_EXTERNAL_OBJECTS =

vseg: CMakeFiles/vseg.dir/vseg.cc.o
vseg: CMakeFiles/vseg.dir/build.make
vseg: /usr/local/lib/libopencv_stitching.so.3.4.7
vseg: /usr/local/lib/libopencv_superres.so.3.4.7
vseg: /usr/local/lib/libopencv_videostab.so.3.4.7
vseg: /usr/local/lib/libopencv_aruco.so.3.4.7
vseg: /usr/local/lib/libopencv_bgsegm.so.3.4.7
vseg: /usr/local/lib/libopencv_bioinspired.so.3.4.7
vseg: /usr/local/lib/libopencv_ccalib.so.3.4.7
vseg: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.7
vseg: /usr/local/lib/libopencv_dpm.so.3.4.7
vseg: /usr/local/lib/libopencv_face.so.3.4.7
vseg: /usr/local/lib/libopencv_freetype.so.3.4.7
vseg: /usr/local/lib/libopencv_fuzzy.so.3.4.7
vseg: /usr/local/lib/libopencv_hfs.so.3.4.7
vseg: /usr/local/lib/libopencv_img_hash.so.3.4.7
vseg: /usr/local/lib/libopencv_line_descriptor.so.3.4.7
vseg: /usr/local/lib/libopencv_optflow.so.3.4.7
vseg: /usr/local/lib/libopencv_reg.so.3.4.7
vseg: /usr/local/lib/libopencv_rgbd.so.3.4.7
vseg: /usr/local/lib/libopencv_saliency.so.3.4.7
vseg: /usr/local/lib/libopencv_stereo.so.3.4.7
vseg: /usr/local/lib/libopencv_structured_light.so.3.4.7
vseg: /usr/local/lib/libopencv_surface_matching.so.3.4.7
vseg: /usr/local/lib/libopencv_tracking.so.3.4.7
vseg: /usr/local/lib/libopencv_xfeatures2d.so.3.4.7
vseg: /usr/local/lib/libopencv_ximgproc.so.3.4.7
vseg: /usr/local/lib/libopencv_xobjdetect.so.3.4.7
vseg: /usr/local/lib/libopencv_xphoto.so.3.4.7
vseg: /usr/local/lib/libopencv_shape.so.3.4.7
vseg: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.7
vseg: /usr/local/lib/libopencv_video.so.3.4.7
vseg: /usr/local/lib/libopencv_datasets.so.3.4.7
vseg: /usr/local/lib/libopencv_plot.so.3.4.7
vseg: /usr/local/lib/libopencv_text.so.3.4.7
vseg: /usr/local/lib/libopencv_dnn.so.3.4.7
vseg: /usr/local/lib/libopencv_highgui.so.3.4.7
vseg: /usr/local/lib/libopencv_ml.so.3.4.7
vseg: /usr/local/lib/libopencv_videoio.so.3.4.7
vseg: /usr/local/lib/libopencv_imgcodecs.so.3.4.7
vseg: /usr/local/lib/libopencv_objdetect.so.3.4.7
vseg: /usr/local/lib/libopencv_calib3d.so.3.4.7
vseg: /usr/local/lib/libopencv_features2d.so.3.4.7
vseg: /usr/local/lib/libopencv_flann.so.3.4.7
vseg: /usr/local/lib/libopencv_photo.so.3.4.7
vseg: /usr/local/lib/libopencv_imgproc.so.3.4.7
vseg: /usr/local/lib/libopencv_core.so.3.4.7
vseg: CMakeFiles/vseg.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/i82carij/3º/FSIV_en/Practical_assignment/backg_moodle_v01/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable vseg"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vseg.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/vseg.dir/build: vseg

.PHONY : CMakeFiles/vseg.dir/build

CMakeFiles/vseg.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/vseg.dir/cmake_clean.cmake
.PHONY : CMakeFiles/vseg.dir/clean

CMakeFiles/vseg.dir/depend:
	cd /home/i82carij/3º/FSIV_en/Practical_assignment/backg_moodle_v01/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/i82carij/3º/FSIV_en/Practical_assignment/backg_moodle_v01 /home/i82carij/3º/FSIV_en/Practical_assignment/backg_moodle_v01 /home/i82carij/3º/FSIV_en/Practical_assignment/backg_moodle_v01/build /home/i82carij/3º/FSIV_en/Practical_assignment/backg_moodle_v01/build /home/i82carij/3º/FSIV_en/Practical_assignment/backg_moodle_v01/build/CMakeFiles/vseg.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/vseg.dir/depend

