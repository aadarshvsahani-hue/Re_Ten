# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
# Note: This assumes your Flask application is in a file named app.py
# and that your Flask app run command is 'python app.py'
# In a real scenario, you might use an entrypoint script or Gunicorn/uWSGI
# If your API code is in a different file, adjust 'app.py' accordingly.
CMD ["python", "app.py"]
