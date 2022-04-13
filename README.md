# CMPUT 656 Election Act Parser

Some code snippets come from documentation (e.g. loading from tensorflow-hub or spacy), and those are without citation. 
There is one snippet that comes from stackoverflow that is marked in the comments. 
Most of the code is custom written for the project. 

All code output is in the notebooks for the project. 
The notebook `replication book` will guide the use of the classes and functions, while the notebook `running` has a more complete guide of how the code was used to produce the results in the report.
The output in replication book was reproduced after the original reporting was complete.

## Running Code
### requires 
* Docker
* Docker-compose

To create the environment, clone this repository and run `docker-compose up --build` which will spawn a docker container with all dependancies in the current directory loaded in /work
Open the chosen notebook in the jupyter server and it should execute. 

if docker is not used, requirements used in the image for the project are in `requirements.txt`, based off a docker, tensorflow-gpu image. If desired, install with `pip install -r requirements.txt`, though your mileage may vary. 


