function NetworkGraph({ graph }) {
  if (!graph) return null;

  return (
    <div style={{ marginTop: "40px" }}>
      <h2>Company Network</h2>

      <pre
        style={{
          background: "#111",
          padding: "20px",
          overflow: "auto",
        }}
      >
        {JSON.stringify(
          graph,
          null,
          2
        )}
      </pre>
    </div>
  );
}

export default NetworkGraph;