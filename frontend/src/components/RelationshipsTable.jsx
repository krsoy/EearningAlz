function RelationshipsTable({
  relationships,
}) {
  if (
    !relationships ||
    relationships.length === 0
  ) {
    return null;
  }

  const columns =
    Object.keys(
      relationships[0]
    );

  return (
    <div
      style={{
        marginTop: "40px",
      }}
    >
      <h2>
        Relationships
      </h2>

      <table
        style={{
          width: "100%",
          borderCollapse:
            "collapse",
        }}
      >
        <thead>
          <tr>
            {columns.map(
              (column) => (
                <th
                  key={column}
                >
                  {column}
                </th>
              )
            )}
          </tr>
        </thead>

        <tbody>
          {relationships.map(
            (
              row,
              index
            ) => (
              <tr
                key={index}
              >
                {columns.map(
                  (
                    column
                  ) => (
                    <td
                      key={
                        column
                      }
                    >
                      {
                        row[
                          column
                        ]
                      }
                    </td>
                  )
                )}
              </tr>
            )
          )}
        </tbody>
      </table>
    </div>
  );
}

export default RelationshipsTable;