function EventsTable({ events }) {
  if (!events || events.length === 0) {
    return null;
  }

  const columns = Object.keys(events[0]);

  return (
    <div style={{ marginTop: "40px" }}>
      <h2>Propagation Events</h2>

      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
        }}
      >
        <thead>
          <tr>
            {columns.map((column) => (
              <th
                key={column}
                style={{
                  borderBottom: "1px solid #444",
                  padding: "8px",
                  textAlign: "left",
                }}
              >
                {column}
              </th>
            ))}
          </tr>
        </thead>

        <tbody>
          {events.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td
                  key={column}
                  style={{
                    padding: "6px",
                  }}
                >
                  {String(row[column])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default EventsTable;