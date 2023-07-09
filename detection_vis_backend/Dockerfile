FROM python:3.10.1-slim

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' detect-vis-api-user

WORKDIR /opt

COPY . .
# ADD data /opt/data
# ADD alembic /opt/alembic
# ADD alembic.ini /opt/alembic.ini
# ADD detection_vis_backend /opt/detection_vis_backend

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /opt/detection_vis_backend/requirements.txt

RUN chmod +x /opt/detection_vis_backend/run.sh
RUN chown -R detect-vis-api-user:detect-vis-api-user ./

USER detect-vis-api-user

EXPOSE 8001

CMD ["bash", "detection_vis_backend/run.sh"]
