function StatsCards({ summary }) {
  return (
    <div
      style={{
        display: "flex",
        gap: "20px",
        marginTop: "20px",
      }}
    >
      <div
        style={{
          border: "1px solid #333",
          padding: "20px",
          borderRadius: "10px",
          minWidth: "200px",
        }}
      >
        <h3>Relationships</h3>
        <h2>{summary.relationships.toLocaleString()}</h2>
      </div>

      <div
        style={{
          border: "1px solid #333",
          padding: "20px",
          borderRadius: "10px",
          minWidth: "200px",
        }}
      >
        <h3>Events</h3>
        <h2>{summary.events.toLocaleString()}</h2>
      </div>

      <div
        style={{
          border: "1px solid #333",
          padding: "20px",
          borderRadius: "10px",
          minWidth: "200px",
        }}
      >
        <h3>Statistics</h3>
        <h2>{summary.stats.toLocaleString()}</h2>
      </div>
    </div>
  );
}

export default StatsCards;