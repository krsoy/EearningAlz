function SectionCard({
  title,
  children,
}) {
  return (
    <div
      style={{
        border:
          "1px solid #333",
        borderRadius:
          "10px",
        padding:
          "20px",
        marginTop:
          "20px",
      }}
    >
      <h2>
        {title}
      </h2>

      {children}
    </div>
  );
}

export default SectionCard;