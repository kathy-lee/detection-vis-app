FROM python:3.10.1-slim

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' detect-vis-api-user

WORKDIR /opt

ADD data /opt/data
ADD detection-vis-backend /opt/detection-vis-backend
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /opt/detection-vis-backend/requirements.txt

RUN chmod +x /opt/detection-vis-backend/run.sh
RUN chown -R detect-vis-api-user:detect-vis-api-user ./

USER detect-vis-api-user

EXPOSE 8001

CMD ["bash", "detection-vis-backend/run.sh"]
