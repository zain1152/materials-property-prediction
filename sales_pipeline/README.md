# Automated Sales Data Pipeline & Interactive Analytics Dashboard

An end-to-end data engineering and analytics pipeline that automates the extraction, transformation, and loading (ETL) of e-commerce data from local source files into a PostgreSQL database, paired with an interactive Power BI dashboard for executive insights.

## 📌 Project Overview
Manual data updates slow down business decision-making and increase the risk of operational errors. This project replaces manual workflows with an automated, transaction-safe Python ETL pipeline that seamlessly synchronizes operational data into pgAdmin/PostgreSQL without interrupting downstream analytics dependencies.

### 📊 Executive Dashboard View
![Power BI Dashboard Overview](images/dashboard_screenshot.png)

---

## 🛠️ Tech Stack & Architecture
* **Language:** Python 3.12+
* **Data Manipulation:** Pandas
* **Database Engine & Driver:** PostgreSQL, SQLAlchemy, Psycopg2
* **Database Management:** pgAdmin 4
* **Business Intelligence:** Power BI Desktop
* **Version Control:** Git & GitHub

### 🔄 The ETL Pipeline Logic
1. **Environment Security:** Loads isolated, encrypted database credentials dynamically via a local `.env` file wrapper to prevent security leaks.
2. **Automated Schema Discovery:** Maps physical relational entities (`categories`, `customers`, `products`, `orders`, `order_items`) directly to target tables.
3. **Transaction-Safe Sync:** Utilizes an isolated database connection sequence to run `TRUNCATE ... CASCADE` routines. This clears transaction records immediately while leaving complex downstream reporting structures (like SQL Views) fully intact.
4. **Batch Stream Loading:** Streams new data rows into target database endpoints using highly efficient, chunked insertions.

---

## 📂 Repository Structure
```text
├── D:\sales_pipeline\
│   ├── images/
│   │   └── dashboard_screenshot.png  # Dashboard visuals for documentation
│   ├── Automation_2.py               # Main Python ETL pipeline script
│   ├── category.sql                  # Database table definition schemas
│   ├── customers.sql                 
│   ├── orders.sql                    
│   ├── order_items.sql               
│   ├── products.sql                  
│   ├── .gitignore                    # Prevents credentials from leaking to GitHub
│   └── README.md                     # Project documentation