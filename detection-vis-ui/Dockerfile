FROM python:3.10.1-slim

WORKDIR /opt/detection-vis-ui


ADD ./detection-vis-ui /opt/detection-vis-ui
RUN pip install --upgrade pip
RUN pip install -r /opt/detection-vis-ui/requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]