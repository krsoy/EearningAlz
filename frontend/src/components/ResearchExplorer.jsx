import PropagationEventsTable
from "./PropagationEventsTable";

function ResearchExplorer({
  events,
}) {
  if (!events?.length)
    return null;

  return (
    <div
      style={{
        marginTop:
          "40px",
      }}
    >
      <h2>
        Research Explorer
      </h2>

      <PropagationEventsTable
        events={events}
      />
    </div>
  );
}

export default ResearchExplorer;