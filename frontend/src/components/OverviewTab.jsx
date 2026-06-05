import CompanyExplorer from "./CompanyExplorer";

function OverviewTab({
  company,
  relationships,
}) {
  return (
    <CompanyExplorer
      company={company}
      relationships={
        relationships
      }
    />
  );
}

export default OverviewTab;