// src/services/api.js

import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8000/earningalz",
});

export const getSummary = async () => {
  const response = await API.get("/summary");
  return response.data;
};

export const getTopSignals = async () => {
  const response = await API.get("/top-signals");
  return response.data;
};

export const getNetwork = async (ticker) => {
  const response = await API.get(`/network/${ticker}`);
  return response.data;
};

export const getCompany = async (ticker) => {
  const response = await API.get(`/company/${ticker}`);
  return response.data;
};

export const getRelationships = async (ticker, limit = 100) => {
  const response = await API.get(
    `/company/${ticker}/relationships?limit=${limit}`
  );
  return response.data;
};

export const getEvents = async (ticker, limit = 100) => {
  const response = await API.get(
    `/company/${ticker}/events?limit=${limit}`
  );
  return response.data;
};