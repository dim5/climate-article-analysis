namespace Data.DTO
{
    using Newtonsoft.Json;
    using Newtonsoft.Json.Converters;
    using System;
    using System.Globalization;

    public partial class Extract
    {
        [JsonProperty("title")]
        public string Title { get; set; }

        [JsonProperty("content")]
        public string Content { get; set; }

        [JsonProperty("textContent")]
        public string TextContent { get; set; }

        [JsonProperty("length")]
        public long Length { get; set; }

        [JsonProperty("siteName")]
        public string SiteName { get; set; }

        [JsonProperty("url")]
        public Uri Url { get; set; }
    }

    public partial class Extract
    {
        public static Extract FromJson(string json) => JsonConvert.DeserializeObject<Extract>(json, Converter.Settings);
    }

    public static class Serialize
    {
        public static string ToJson(this Extract self) => JsonConvert.SerializeObject(self, Converter.Settings);
    }

    internal static class Converter
    {
        public static readonly JsonSerializerSettings Settings = new JsonSerializerSettings
        {
            MetadataPropertyHandling = MetadataPropertyHandling.Ignore,
            DateParseHandling = DateParseHandling.None,
            Converters =
            {
                new IsoDateTimeConverter { DateTimeStyles = DateTimeStyles.AssumeUniversal }
            },
        };
    }
}