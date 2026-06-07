import { useState } from "react";

function CompanySearch({
  onSearch,
}) {
  const [ticker, setTicker] =
    useState("");

  const handleSubmit = (
    e
  ) => {
    e.preventDefault();

    if (
      !ticker.trim()
    )
      return;

    onSearch(
      ticker.toUpperCase()
    );
  };

  return (
    <form
      onSubmit={
        handleSubmit
      }
      style={{
        marginTop:
          "40px",
      }}
    >
      <input
        type="text"
        placeholder="AAPL"
        value={ticker}
        onChange={(e) =>
          setTicker(
            e.target.value
          )
        }
      />

      <button
        type="submit"
      >
        Search
      </button>
    </form>
  );
}

export default CompanySearch;