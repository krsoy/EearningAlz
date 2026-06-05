function SignalExplorer({
  signals,
}) {
  if (!signals?.length)
    return null;

  return (
    <div
      style={{
        marginTop:
          "40px",
      }}
    >
      <h2>
        Signal Explorer
      </h2>

      <table
        style={{
          width: "100%",
        }}
      >
        <thead>
          <tr>
            <th>
              Signal
            </th>

            <th>
              Relation
            </th>

            <th>
              Accuracy
            </th>

            <th>
              Edges
            </th>
          </tr>
        </thead>

        <tbody>
          {signals.map(
            (
              row,
              index
            ) => (
              <tr
                key={index}
              >
                <td>
                  {
                    row.signal
                  }
                </td>

                <td>
                  {
                    row.relation_group
                  }
                </td>

                <td>
                  {(
                    row.prediction_accuracy *
                    100
                  ).toFixed(
                    1
                  )}
                  %
                </td>

                <td>
                  {
                    row.exposed_edges
                  }
                </td>
              </tr>
            )
          )}
        </tbody>
      </table>
    </div>
  );
}

export default SignalExplorer;