# Fashion Mnist

## Installation Guideline

```
git clone https://github.com/Anuj040/fashion_mnist.git [-b <branch_name>]
cd fashion_mnist (Work Directory)

# local environment settings
pyenv local 3.8.5 (^3.8)
python -m pip install poetry
poetry config virtualenvs.create true --local
poetry config virtualenvs.in-project true --local

# In case older version of pip throws installation errors
poetry run python -m pip install --upgrade pip 

# local environment preparation
poetry install
```

## Launching the API
Before running the API we need to build the docker image and run the container

### Building the docker imge 
* For building the image please execute the following from the work directory
```
docker build -t mnist/api . 
```
* You can also let the Makefile do this for you using the following
```
make docker_build
```
It will take sometime, so please grab a coffee or any drink of your choice.

*Note: The included Dockerfile uses tensorflow/tensorflow:2.5.0-gpu as the base image.*
*If you plan to work on a cpu-only machine please consider using tensorflow/tensorflow:2.5.0.*

### Running the docker container
* Once the docker image is done building, please run the docker container with the following command fromt the terminal.
```
docker run -d -p 5000:5000 mnist/api
```
* from the Makefile
```
make docker_run
```
If everything has gone as per the plan, the webAPI should be accessible through http://localhost:5000/

## Working with the API
The API has three different functionalities. 

1. Model training
  * Accessing http://localhost:5000/train should land you at the following page.
  ![image](https://user-images.githubusercontent.com/66895104/121269361-a923b180-c8fa-11eb-8faa-ca5ccebc691c.png)
  * One can chose the number of _epohcs_, _batch_size_ and _learning_rate_ for model training.
  * Model is trained on the train split of _mnist_fashion_ dataset. 
  * At the end of the training, you should be redirected to the following page.
  ![image](https://user-images.githubusercontent.com/66895104/121270014-d6249400-c8fb-11eb-905f-cb173ddb2e9a.png)
 **Note: To track the model performance during training, by default the current implementation, splits the original training set of _mnist fashion_
 into train/val (80:20) split for classifier training.

 2. Model Evaluation
 * Clicking on _eval_ in the above page should redirect you to model evaluation page.
 * It can also be directly accessed from http://localhost:5000/eval
 ![image](https://user-images.githubusercontent.com/66895104/121270308-6a8ef680-c8fc-11eb-8c13-6f4737f626eb.png)
 * Clicking the _execute_ button will start model evaluation on the test split of _mnist fashion_ dataset. 
 * At the end of evaluation step, it should land you on the following page.
 [image](https://user-images.githubusercontent.com/66895104/121270756-44b62180-c8fd-11eb-9ca6-64e8e5046ffa.png)

  3. Model Evaluation
  * Lastly to test the model you can go to http://localhost:5000/infer
  ![image](https://user-images.githubusercontent.com/66895104/121270887-834bdc00-c8fd-11eb-9bb1-8d320c766df5.png)
  * From the _choose_file_ option please select an image to be tested and click _submit_. For example, the following image
  ![image](https://user-images.githubusercontent.com/66895104/121271279-4d5b2780-c8fe-11eb-9132-9ce8fba76665.png)
  * The API should return you with a response in the following format.
  ![image](https://user-images.githubusercontent.com/66895104/121271167-0ec56d00-c8fe-11eb-911e-64603b447ec2.png)

**Note: Make sure that the image to be tested matches the _mnist fashion_ dataset dimensions, i.e., a single channel image of size (28, 28).
