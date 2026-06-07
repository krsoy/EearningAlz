function DashboardTabs({
  activeTab,
  setActiveTab,
}) {
  const tabs = [
    "overview",
    "research",
    "network",
  ];

  return (
    <div
      style={{
        display: "flex",
        gap: "10px",
        marginTop: "20px",
        marginBottom: "20px",
      }}
    >
      {tabs.map((tab) => (
        <button
          key={tab}
          onClick={() =>
            setActiveTab(tab)
          }
          style={{
            padding: "10px 20px",
            cursor: "pointer",
            fontWeight:
              activeTab === tab
                ? "bold"
                : "normal",
          }}
        >
          {tab}
        </button>
      ))}
    </div>
  );
}

export default DashboardTabs;