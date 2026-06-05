function NetworkList({
  graph,
}) {
  if (!graph)
    return null;

  return (
    <div
      style={{
        marginTop:
          "40px",
      }}
    >
      <h2>
        Network
      </h2>

      <ul>
        {graph.nodes.map(
          (node) => (
            <li
              key={
                node.id
              }
            >
              {
                node.id
              }
            </li>
          )
        )}
      </ul>
    </div>
  );
}

export default NetworkList;