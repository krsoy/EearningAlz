import { useEffect, useState } from "react";

import {
  getSummary,
  getTopSignals,
  getCompany,
  getRelationships,
  getEvents,
  getNetwork,
} from "../services/api";

import DashboardHeader from "../components/DashboardHeader";
import DashboardTabs from "../components/DashboardTabs";

import StatsCards from "../components/StatsCards";
import TopSignalsTable from "../components/TopSignalsTable";

import CompanySearch from "../components/CompanySearch";

import OverviewTab from "../components/OverviewTab";
import ResearchTab from "../components/ResearchTab";
import NetworkTab from "../components/NetworkTab";

import FilterBar from "../components/FilterBar";

function Dashboard() {
  const [summary, setSummary] =
    useState(null);

  const [signals, setSignals] =
    useState([]);

  const [company, setCompany] =
    useState(null);

  const [relationships, setRelationships] =
    useState([]);

  const [events, setEvents] =
    useState([]);

  const [graph, setGraph] =
    useState(null);

  const [activeTab, setActiveTab] =
    useState("overview");

  const [signalFilter, setSignalFilter] =
    useState("");

  useEffect(() => {
    const loadData = async () => {
      try {
        const summaryData =
          await getSummary();

        setSummary(summaryData);

        const signalData =
          await getTopSignals();

        setSignals(signalData);
      } catch (error) {
        console.error(error);
      }
    };

    loadData();
  }, []);

  const searchCompany =
    async (ticker) => {
      try {
        const companyData =
          await getCompany(
            ticker
          );

        setCompany(
          companyData
        );

        const relationshipData =
          await getRelationships(
            ticker
          );

        setRelationships(
          relationshipData
        );

        const eventData =
          await getEvents(
            ticker
          );

        setEvents(
          eventData
        );

        const networkData =
          await getNetwork(
            ticker
          );

        setGraph(
          networkData
        );
      } catch (error) {
        console.error(error);
      }
    };

  if (!summary) {
    return (
      <div>
        Loading...
      </div>
    );
  }

  const filteredEvents =
    signalFilter === ""
      ? events
      : events.filter(
          (event) =>
            event.signal
              ?.toLowerCase()
              .includes(
                signalFilter.toLowerCase()
              )
        );

  return (
    <div
      style={{
        padding: "20px",
        maxWidth: "1600px",
        margin: "0 auto",
      }}
    >
      <DashboardHeader />

      <StatsCards
        summary={summary}
      />

      <TopSignalsTable
        signals={signals}
      />

      <CompanySearch
        onSearch={
          searchCompany
        }
      />

      <DashboardTabs
        activeTab={
          activeTab
        }
        setActiveTab={
          setActiveTab
        }
      />

      {activeTab ===
        "research" && (
        <FilterBar
          signal={
            signalFilter
          }
          setSignal={
            setSignalFilter
          }
        />
      )}

      {activeTab ===
        "overview" && (
        <OverviewTab
          company={company}
          relationships={
            relationships
          }
        />
      )}

      {activeTab ===
        "research" && (
        <ResearchTab
          events={
            filteredEvents
          }
        />
      )}

      {activeTab ===
        "network" && (
        <NetworkTab
          graph={graph}
        />
      )}
    </div>
  );
}

export default Dashboard;