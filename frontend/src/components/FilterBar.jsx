function FilterBar({
  signal,
  setSignal,
}) {
  return (
    <div
      style={{
        marginTop: "20px",
        marginBottom: "20px",
      }}
    >
      <input
        type="text"
        placeholder="Filter signal..."
        value={signal}
        onChange={(e) =>
          setSignal(
            e.target.value
          )
        }
      />
    </div>
  );
}

export default FilterBar;