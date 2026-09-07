## Prerequisites
Ensure the host computer has the following software installed:
- [Docker Desktop](https://www.docker.com/get-started/) (or Docker Engine + Docker Compose on Linux)
- [Git](https://git-scm.com/install/)

## Setup
### Clone this repository
Open a terminal and run the command below:
```bash
git clone https://github.com/leticiaarco/ELS-PREDIN-APP.git

# or

git clone git@github.com:leticiaarco/ELS-PREDIN-APP.git
```

Alternatively, you can download the project as a zip file as follow: 
- Click the green `<> Code` button, then
- Click the `Download Zip` option
- Unzip the downloaded file and follow the next steps

### Launch Services
Open the local directory `ELS-PREDIN-APP` in a terminal using the command below:
```bash
cd PATH-TO_ELS-PREDIN-APP
```

Then run the `compose` command to build and launch all the services:
```bash
docker compose up -d --build
```

*What happens during this step:*
- Docker builds the Flask web application container.
- Docker downloads the official Ollama container.
- The entrypoint.sh script automatically starts Ollama and downloads the required
language model (qwen2.5:1.5b).

### Check startup logs
The initial run requires downloading the AI model (approx. 1.0 GB). To check if the model
download is finished:
```bash
docker compose logs -f ollama
```
Look for the line: Ollama setup complete! System ready. Once you see this message, press `Ctrl + C` to exit the log viewer.

## Application Usage
Once the system is ready, open any web browser and visit: `http://localhost:5000`. 
You can now upload data, view predictions, and generate AI risk summaries.

<img width="1336" height="1084" alt="image" src="https://github.com/user-attachments/assets/0cdc4a0b-8567-4b09-bf07-be934c006d78" />


## Daily Usage Commands

### Stopping the application
To temporarily stop the application and free up computer memory, run the command below:
```bash
docker compose stop
```

### Start the application
```bash
docker compose start
```

## Shutdown & Cleanup Options
Depending on your maintenance needs, use one of the following commands:

*Option A: Standard Shutdown (Recommended)*
Stops and removes active containers, but keeps downloaded AI models saved on disk
for fast restarts next time:
```bash
docker compose down
```

*Option B: Complete Reset & Cleanup*
Deletes containers, network rules, and all downloaded AI model files (requiring a full
redownload on the next launch):
```bash
docker compose down -v --rmi all --remove-orphans
```
