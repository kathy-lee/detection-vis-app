FROM python:3.10.1-slim

WORKDIR /opt/detection_vis_ui


ADD ./detection_vis_ui /opt/detection_vis_ui
RUN pip install --upgrade pip
RUN pip install -r /opt/detection_vis_ui/requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]