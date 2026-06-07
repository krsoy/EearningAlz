import {
  useMemo,
} from "react";

function EventFilters({
  events,
  signal,
  setSignal,
}) {
  const signals =
    useMemo(() => {
      const unique =
        [
          ...new Set(
            events.map(
              (
                e
              ) =>
                e.signal
            )
          ),
        ];

      return unique.sort();
    }, [events]);

  return (
    <div
      style={{
        marginTop:
          "20px",
      }}
    >
      <label>
        Signal:
      </label>

      <select
        value={signal}
        onChange={(e) =>
          setSignal(
            e.target.value
          )
        }
      >
        <option value="">
          All
        </option>

        {signals.map(
          (s) => (
            <option
              key={s}
              value={s}
            >
              {s}
            </option>
          )
        )}
      </select>
    </div>
  );
}

export default EventFilters;