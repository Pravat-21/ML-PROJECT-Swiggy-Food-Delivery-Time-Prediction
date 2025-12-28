# set the base image
FROM python:3.12-slim

# install lightgbm dependency
RUN apt-get update && apt-get install -y libgomp1

# set up the working directory
WORKDIR /app

#copy my fastapi_app folder
COPY fastapi_app/ /app/

# install the packages
RUN pip install -r requirements.txt

# copy the app contents
COPY ./models/processor.pkl ./models/processor.pkl
COPY ./src/logger.py ./src/logger.py
COPY ./src/utils.py ./src/utils.py
COPY ./src/exception.py ./src/exception.py
COPY ./reports/models_info.json ./reports/models_info.json

# expose the port
EXPOSE 8000

# Run the file using command
CMD [ "python","./app.py" ]