using System.Text;
using System.Text.RegularExpressions;

namespace MarkdownChunking;

public enum BlockType { Heading, Paragraph, Table, List, Other }

/// <summary>
/// One structural unit from the markdown, in reading order. You produce these from the
/// Document Intelligence markdown (e.g. via Markdig). The chunker derives all section
/// structure from <see cref="HeadingLevel"/> alone — nothing here is domain-specific.
/// </summary>
public sealed class MarkdownBlock
{
    public required BlockType Type { get; init; }

    /// <summary>Verbatim markdown for this block — this is what ends up embedded.</summary>
    public required string RawText { get; init; }

    /// <summary>1..6 for a heading, 0 otherwise.</summary>
    public int HeadingLevel { get; init; }

    /// <summary>Plain heading text (optional; derived from RawText if left empty).</summary>
    public string HeadingTitle { get; init; } = "";

    /// <summary>1-based source page, if known (for mapping back to bounding boxes).</summary>
    public int? Page { get; init; }
}

/// <summary>An embed-ready chunk. Maps onto your existing index; <see cref="Content"/> is the text field.</summary>
public sealed class SectionChunk
{
    public required string DocumentId { get; init; }
    public required string ChunkId { get; init; }

    /// <summary>Full heading breadcrumb, e.g. "Top heading › Sub heading".</summary>
    public required string SectionPath { get; init; }

    /// <summary>The nearest boundary-level heading that owns this chunk.</summary>
    public string SectionTitle { get; init; } = "";

    public int? StartPage { get; init; }
    public int? EndPage { get; init; }
    public int TokenCount { get; init; }

    public required string Content { get; init; }

    /// <summary>Optional facet (= <see cref="SectionTitle"/>) for a hard $filter at query time.</summary>
    public string? SectionKey { get; init; }

    public float[]? Vector { get; set; }
}

public sealed class ChunkingOptions
{
    public int MaxTokens { get; init; } = 2000;
    public int OverlapTokens { get; init; } = 200;

    /// <summary>
    /// Headings at this level or shallower start a new chunk group; chunks never cross them.
    /// Deeper headings only refine the breadcrumb. With no headings present, the whole document
    /// is one group (plain token windowing — graceful fallback for flat documents).
    /// </summary>
    public int SectionBoundaryLevel { get; init; } = 2;

    public string DocumentTitle { get; init; } = "";

    /// <summary>Populate <see cref="SectionChunk.SectionKey"/> for filtering.</summary>
    public bool EmitSectionKey { get; init; } = false;

    public string PathSeparator { get; init; } = " › ";
}

public interface ITokenCounter { int Count(string text); }

/// <summary>
/// Generic, heading-driven chunker. Two passes:
///   1. Walk blocks, maintaining a heading stack, and assign each block a section path + group id.
///   2. Within each group, greedily token-window (with overlap), keep tables atomic, and prefix
///      each chunk with its section breadcrumb so the identity is embedded into the text itself.
/// </summary>
public sealed class MarkdownSectionChunker
{
    private static readonly Regex SentenceSplit = new(@"(?<=[.!?])\s+", RegexOptions.Compiled);
    private static readonly Regex HeadingMarkup = new(@"^\s*#{1,6}\s*", RegexOptions.Compiled);

    private readonly ChunkingOptions _opt;
    private readonly ITokenCounter _tok;

    public MarkdownSectionChunker(ChunkingOptions options, ITokenCounter tokenCounter)
    {
        _opt = options;
        _tok = tokenCounter;
    }

    public IReadOnlyList<SectionChunk> Chunk(string documentId, IReadOnlyList<MarkdownBlock> blocks)
    {
        var annotated = AssignSections(blocks);
        var output = new List<SectionChunk>();

        foreach (var group in GroupBySection(annotated))
        {
            var seq = 0;
            var carryOver = "";
            var current = new List<Annotated>();
            var currentTokens = 0;

            void Flush()
            {
                if (current.Count == 0) return;
                output.Add(Build(documentId, current, carryOver, seq++));
                carryOver = TailByTokens(BodyText(current), _opt.OverlapTokens);
                current.Clear();
                currentTokens = 0;
            }

            foreach (var a in group)
            {
                var blockTokens = _tok.Count(a.Block.RawText);

                // Atomic oversize table: flush, emit alone, no overlap.
                if (a.Block.Type == BlockType.Table && blockTokens > _opt.MaxTokens)
                {
                    Flush();
                    output.Add(Build(documentId, new[] { a }, "", seq++));
                    carryOver = "";
                    continue;
                }

                if (currentTokens + blockTokens > _opt.MaxTokens && current.Count > 0)
                    Flush();

                current.Add(a);
                currentTokens += blockTokens;
            }

            Flush();
        }

        return output;
    }

    // ---- pass 1: section assignment from heading levels ----

    private readonly record struct Annotated(MarkdownBlock Block, string Path, string Title, int GroupId);

    private List<Annotated> AssignSections(IReadOnlyList<MarkdownBlock> blocks)
    {
        var result = new List<Annotated>(blocks.Count);
        var stack = new List<(int Level, string Title)>();
        var groupId = 0;
        var started = false;

        foreach (var b in blocks)
        {
            if (b.Type == BlockType.Heading && b.HeadingLevel > 0)
            {
                // Pop siblings/deeper headings, then push this one.
                while (stack.Count > 0 && stack[^1].Level >= b.HeadingLevel)
                    stack.RemoveAt(stack.Count - 1);

                var title = HeadingTitle(b);
                stack.Add((b.HeadingLevel, title));

                // A heading at/above the boundary level opens a new chunk group.
                if (b.HeadingLevel <= _opt.SectionBoundaryLevel)
                {
                    if (started) groupId++;
                    started = true;
                }
            }

            var path = string.Join(_opt.PathSeparator, stack.Select(s => s.Title));
            var sectionTitle = NearestBoundaryTitle(stack);
            result.Add(new Annotated(b, path, sectionTitle, groupId));
        }

        return result;
    }

    private string NearestBoundaryTitle(List<(int Level, string Title)> stack)
    {
        for (var i = stack.Count - 1; i >= 0; i--)
            if (stack[i].Level <= _opt.SectionBoundaryLevel)
                return stack[i].Title;
        return stack.Count > 0 ? stack[^1].Title : "";
    }

    private static string HeadingTitle(MarkdownBlock b) =>
        !string.IsNullOrWhiteSpace(b.HeadingTitle)
            ? b.HeadingTitle.Trim()
            : HeadingMarkup.Replace(b.RawText, "").Trim();

    // ---- pass 2 helpers ----

    private static IEnumerable<List<Annotated>> GroupBySection(List<Annotated> blocks)
    {
        var runs = new List<List<Annotated>>();
        foreach (var a in blocks)
        {
            if (runs.Count == 0 || runs[^1][0].GroupId != a.GroupId)
                runs.Add(new List<Annotated>());
            runs[^1].Add(a);
        }
        return runs;
    }

    private static string BodyText(IReadOnlyList<Annotated> items) =>
        string.Join("\n\n", items.Select(i => i.Block.RawText.Trim()));

    private SectionChunk Build(string documentId, IReadOnlyList<Annotated> items, string carryOver, int seq)
    {
        var body = BodyText(items);
        if (!string.IsNullOrEmpty(carryOver))
            body = carryOver + "\n\n" + body;

        var first = items[0];
        var content = BuildPrefix(first.Path) + "\n\n" + body;

        var pages = items.Where(i => i.Block.Page is not null).Select(i => i.Block.Page!.Value).ToList();
        var slug = Slug(string.IsNullOrEmpty(first.Title) ? "section" : first.Title);

        return new SectionChunk
        {
            DocumentId = documentId,
            ChunkId = $"{documentId}_{slug}_{seq:D3}",
            SectionPath = first.Path,
            SectionTitle = first.Title,
            StartPage = pages.Count > 0 ? pages.Min() : null,
            EndPage = pages.Count > 0 ? pages.Max() : null,
            TokenCount = _tok.Count(content),
            Content = content,
            SectionKey = _opt.EmitSectionKey ? first.Title : null
        };
    }

    private string BuildPrefix(string path)
    {
        var sb = new StringBuilder();
        if (!string.IsNullOrWhiteSpace(_opt.DocumentTitle))
            sb.Append($"Document: {_opt.DocumentTitle}.");
        if (!string.IsNullOrWhiteSpace(path))
        {
            if (sb.Length > 0) sb.Append(' ');
            sb.Append($"Section: {path}.");
        }
        return sb.ToString();
    }

    private string TailByTokens(string text, int maxTokens)
    {
        if (maxTokens <= 0 || string.IsNullOrEmpty(text)) return "";
        var sentences = SentenceSplit.Split(text);
        var picked = new List<string>();
        var running = 0;
        for (var i = sentences.Length - 1; i >= 0; i--)
        {
            var t = _tok.Count(sentences[i]);
            if (running + t > maxTokens && picked.Count > 0) break;
            picked.Insert(0, sentences[i]);
            running += t;
        }
        return string.Join(" ", picked).Trim();
    }

    private static string Slug(string s)
    {
        var cleaned = Regex.Replace(s.ToLowerInvariant(), @"[^a-z0-9]+", "-").Trim('-');
        return string.IsNullOrEmpty(cleaned) ? "section" : cleaned;
    }
}
