import CompanyCard from "./CompanyCard";
import RelationshipsTable from "./RelationshipsTable";

function CompanyExplorer({
  company,
  relationships,
}) {
  if (!company) return null;

  return (
    <div>
      <CompanyCard
        company={company}
      />

      <RelationshipsTable
        relationships={
          relationships
        }
      />
    </div>
  );
}

export default CompanyExplorer;