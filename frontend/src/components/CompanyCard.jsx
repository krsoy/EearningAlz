function CompanyCard({
  company,
}) {
  if (!company)
    return null;

  return (
    <div
      style={{
        marginTop:
          "20px",
      }}
    >
      <h2>
        {
          company.ticker
        }
      </h2>

      <p>
        Relationships:
        {" "}
        {
          company.relationships
        }
      </p>

      <p>
        Events as
        Source:
        {" "}
        {
          company.events_as_source
        }
      </p>

      <p>
        Events as
        Target:
        {" "}
        {
          company.events_as_target
        }
      </p>
    </div>
  );
}

export default CompanyCard;