init:
	pip install -r requirements.txt

	cd repos-3rd ; git clone https://github.com/Itseez/opencv.git
	cd repos-3rd/opencv ; git checkout 3.1.0
	cd repos-3rd ; git clone https://github.com/Itseez/opencv_contrib.git
	cd repos-3rd/opencv_contrib ; git checkout 3.1.0
	mkdir repos-3rd/opencv/build
	cd repos-3rd/opencv/build ; cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=../../../venv -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ..
	cd repos-3rd/opencv/build ; make -j4
	cd repos-3rd/opencv/build ; make install

	python setup.py build
install:
	python setup.py install
# test:
# 	py.test tests
