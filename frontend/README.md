# EarningALZ Frontend

## Overview

This frontend provides an interactive interface for exploring the EarningALZ research artifacts.

The application consumes data exposed by the FastAPI backend and allows users to:

* View dataset statistics
* Explore top propagation signals
* Search companies by ticker
* Inspect company relationships
* Explore propagation events
* Navigate network connections

---

## Prerequisites

Before starting the frontend, the backend API must be running.

### Backend

From the project root:

```bash
cd backend

uvicorn app.main:app --reload
```

The backend should be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

## Frontend Installation

Navigate to the frontend folder:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

---

## Running the Frontend

Start the development server:

```bash
npm run dev
```

The frontend will be available at:

```text
http://localhost:5173
```

(or another port shown by Vite)

---

## Current Features

### Dashboard

Displays:

* Relationship statistics
* Event statistics
* Top propagation signals

### Company Explorer

Search companies by ticker:

```text
AAPL
NVDA
MSFT
TSLA
```

Shows:

* Company summary
* Relationships
* Propagation events
* Network information

### Research Explorer

Explore:

* Signal propagation events
* Direction matches
* Prediction correctness

### Network View

Inspect company network connections derived from relationship data.

---

## Architecture

```text
Frontend (React + Vite)
        ↓
FastAPI Backend
        ↓
Parquet Artifacts
        ↓
EarningALZ Research Pipeline
```

---

## Development Notes

Backend must be running before using the frontend.

Current backend endpoints:

```text
GET /earningalz/summary
GET /earningalz/top-signals
GET /earningalz/company/{ticker}
GET /earningalz/company/{ticker}/relationships
GET /earningalz/company/{ticker}/events
GET /earningalz/network/{ticker}
```

---

## Current Project Status

```text
Phase 0 - Repository Archaeology      ✓
Phase 1 - Artifact Mapping            ✓
Phase 2 - Backend MVP                 ✓
Phase 3 - Frontend MVP                ✓

Phase 4 - Research Explorer           In Progress
```
