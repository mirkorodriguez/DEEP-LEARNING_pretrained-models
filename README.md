# Deep Learning

Transfer Learning for Computer Vision deployed on cloud.

#CURL

curl -i -X POST -H "Content-Type: multipart/form-data" -F "file=@image.jpg" http://<public-ip>:5000/inceptionv3/predict/
curl -i -X POST -F "file=@images.jpg" http://<public-ip>:5000/inceptionv3/predict/
