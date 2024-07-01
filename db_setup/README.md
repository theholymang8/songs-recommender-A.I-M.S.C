# Database Setup Guide

## Overview
This README provides detailed instructions on how to set up your MySQL database using a Python script that automates the loading of music-related data. This script populates your MySQL database with data for tracks, albums, and artists. This script populates your MySQL database with data for tracks, albums, and artists derived from the Free Music Archive (FMA).

## Prerequisites
- **MySQL Database**: Ensure that MySQL is installed and running on your system.
- **Python Environment**: You need Python installed, along with the libraries `pandas` and `sqlalchemy`.
- **Data Files**: You should have the CSV files `tracks_parsed.csv`, `albums_parsed.csv`, and `artists_parsed.csv` saved in the same directory as your script.

## Installation

### Python Libraries
Install the required Python libraries using pip if you haven't already (they are also included in the requirements file in the root folder of this repository):

```bash
pip install pandas sqlalchemy mysqlclient
```

## Database Credentials

Edit the script to replace `your_username`, `your_password`, and the database details with your actual MySQL credentials and database name. The default configuration assumes the MySQL server is running locally on port 3306.

Here is a snippet from the script showing the relevant lines:

```python
username = 'your_username'  # Your MySQL username
password = 'your_password'  # Your MySQL password
database = 'multimodal_msc_ai_songs'  # Your database name
```

## Running the Script

- Open your terminal or command prompt.

- Navigate to the directory where your script and CSV files are located.

- Run the script by typing `python database_loader.py`.

The script assumes that the schema will be empty, meaning that it will not contain the tables and specifications of them in the db. Although if you prefer to load them manually, you can find the schema structures in the schemas folder.

## Post-Execution

Once the script has been executed, the data from the CSV files will be loaded into your MySQL database under the tables tracks, albums, and artists. Each table will be created if it does not exist and will replace existing data if it does.

## Troubleshooting

- Connection Errors: Ensure that MySQL is running, and the credentials in the script are correct.
- Data Loading Issues: Check the CSV files for correct encoding and format consistency.
- Library Installation Problems: Verify that all required Python libraries are installed.

## Dataset and Metadata Credits

The data and metadata used in this script are sourced from the Free Music Archive (FMA), a dataset curated by Michaël Defferrard et al. The complete dataset can be found on GitHub and the details of the dataset can be referenced in their research paper:

- Code & Data: FMA GitHub Repository
- Research Paper: FMA: A Dataset For Music Analysis on arXiv

All contributions and rights of the dataset and its metadata belong to the original creators.
