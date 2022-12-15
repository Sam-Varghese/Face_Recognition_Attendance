# FRAS

![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![](https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white)
![](https://img.shields.io/badge/MySQL-005C84?style=for-the-badge&logo=mysql&logoColor=white)


## Face Recognition Attendance System

FRAS is a facial recognition attendance system that aims at simplifying the process of attendance management with the help of state-of-art AI technologies like Tensorflow, Convolutional Neural Networks, etc.

<img src = "https://149695847.v2.pressablecdn.com/wp-content/uploads/2020/10/face.jpg">

---

## Requirements

- Python
- MySQL
- Laptop with a camera

---

## Features

- Automatically captures images of people for facial recognition.
- Integrated with MySQL database which enables this program to update records in real time.
- This program leverages Convolution Neural Networks, Tensorflow, Data Augmentation for the most accurate predictions.
- Model is saved after it's trained everytime which enables the user to re-run the saved model instead of training it again and again on images of all users.
- All the tables in MySQL are well connected and efforts are being taken constantly to reduce redundancies to max extent.
- Entity Relation diagram is also maintained.

---

## How to Use

1. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the repository.
2. [Clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the forked repository.
3. Move into the directory where you cloned to repository.
4. Create a new file namely `connection_info.json`.
5. Put the following details into this file

```json
{
    "mysql_password": "your_mysql_password",
    "mysql_host": "localhost",
    "mysql_user": "root",
    "master_database": "FRAS"
}
```

6. Install the required packages by executing the following command `pip install -r requirements.txt`
7. Now execute [face_recognition_attendance.py](face_recognition_attendance.py) file.

---

## Additional Information

- Project is licensed under [MIT License](./LICENSE)