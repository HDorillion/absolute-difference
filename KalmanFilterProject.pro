TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

unix:LIBS += -L/usr/local/lib -lopencv_imgcodecs \
            -L/usr/local/lib -lopencv_core \
            -L/usr/local/lib -lopencv_highgui \
            -L/usr/local/lib -lopencv_video \
            -L/usr/local/lib -lopencv_videoio \
            -L/usr/local/lib -lopencv_imgproc \
            -L/usr/local/lib -lopencv_features2d \
            -L/usr/local/lib -lopencv_objdetect \
            -I/usr/local/include/libfreenect \

SOURCES += main.cpp
