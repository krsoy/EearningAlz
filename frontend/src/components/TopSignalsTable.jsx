function TopSignalsTable({ signals }) {
  if (!signals || signals.length === 0) {
    return <p>No signals found.</p>;
  }

  return (
    <div style={{ marginTop: "40px" }}>
      <h2>Top Signal Relationships</h2>

      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
        }}
      >
        <thead>
          <tr>
            <th>Signal</th>
            <th>Relationship</th>
            <th>Accuracy</th>
            <th>Edges</th>
          </tr>
        </thead>

        <tbody>
          {signals.map((row, index) => (
            <tr key={index}>
              <td>{row.signal}</td>

              <td>{row.relation_group}</td>

              <td>
                {(row.prediction_accuracy * 100).toFixed(1)}%
              </td>

              <td>
                {row.exposed_edges.toLocaleString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default TopSignalsTable;