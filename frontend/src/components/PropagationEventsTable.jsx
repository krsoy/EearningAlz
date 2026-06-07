function PropagationEventsTable({
  events,
}) {
  if (!events?.length) return null;

  return (
    <div>
      <h2>
        Propagation Events
      </h2>

      <table
        style={{
          width: "100%",
        }}
      >
        <thead>
          <tr>
            <th>Source</th>
            <th>Target</th>
            <th>Signal</th>
            <th>Relation</th>
            <th>Match</th>
            <th>Prediction</th>
          </tr>
        </thead>

        <tbody>
          {events.map(
            (
              event,
              index
            ) => (
              <tr
                key={index}
              >
                <td>
                  {
                    event.source_ticker
                  }
                </td>

                <td>
                  {
                    event.target_ticker
                  }
                </td>

                <td>
                  {
                    event.signal
                  }
                </td>

                <td>
                  {
                    event.relation_group
                  }
                </td>

                <td>
                  {event.direction_match
                    ? "🟢"
                    : "🔴"}
                </td>

                <td>
                  {event.prediction_correct ===
                  true
                    ? "✅"
                    : event.prediction_correct ===
                      false
                    ? "❌"
                    : "-"}
                </td>
              </tr>
            )
          )}
        </tbody>
      </table>
    </div>
  );
}

export default PropagationEventsTable;