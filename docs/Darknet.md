# What is Darknet?
Darknet is a human and object detection library that uses OpenCV and a pre-trained neural network called YOLO (You Only Look Once) to identify humans and objects. We are using a Darknet_wrapper called **darknet_ros**, specifically a fork of it created for ROS2 Humble by [*Ar-Ray-code*](https://github.com/Ar-Ray-code), for our camera visualization and image annotation. 

## How does Darknet work?
Darknet takes an image from a camera, runs the image through YOLO and OpenCV to detect humans and objects, and then outputs annotated bounding boxes around any detected. For this project, we are only having Darknet detect humans by configuring our launch file to ignore anything else, as we do not need to detect anything else.

## How to use Darknet in our environment
- Navigate to the workspace of the docker container `cd /rosEnv`
- Run the "darknet.sh" script to install Darknet `./darknet.sh`
- Source the "build.sh" script `source build.sh`
- To run Darknet, run `ros2 launch darknet_ros stratominers.launch.py`
- Then, in another terminal, navigate to the directory where your bags are placed and play any bag with image data.
- From there, Darknet will launch a window showing the image as well as the annotated bounding boxes.

## References

Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.

    @article{opencv_library,
        author = {Bradski, G.},
        citeulike-article-id = {2236121},
        journal = {Dr. Dobb's Journal of Software Tools},
        keywords = {bibtex-import},
        posted-at = {2008-01-15 19:21:54},
        priority = {4},
        title = {{The OpenCV Library}},
        year = {2000}
    }

S. Macenski, T. Foote, B. Gerkey, C. Lalancette, W. Woodall, “Robot Operating System 2: Design, architecture, and uses in the wild,” Science Robotics vol. 7, May 2022.

    @article{
        doi:10.1126/scirobotics.abm6074,
        author = {Steven Macenski  and Tully Foote  and Brian Gerkey  and Chris Lalancette  and William Woodall },
        title = {Robot Operating System 2: Design, architecture, and uses in the wild},
        journal = {Science Robotics},
        volume = {7},
        number = {66},
        pages = {eabm6074},
        year = {2022},
        doi = {10.1126/scirobotics.abm6074},
        URL = {https://www.science.org/doi/abs/10.1126/scirobotics.abm6074}
    }

Redmon, J. (2013–2016). Darknet: Open Source Neural Networks in C. .

    @misc{darknet13,
        author =   {Redmon, J},
        title =    {Darknet: Open Source Neural Networks in C},
        howpublished = {\url{http://pjreddie.com/darknet/}},
        year = {2013--2016}
    }

Wang, C.Y., Bochkovskiy, A., & Liao, H.Y. (2022). YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. arXiv preprint arXiv:2207.02696.

    @article{wang2022yolov7,
        title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
        author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
        journal={arXiv preprint arXiv:2207.02696},
        year={2022}
    }

